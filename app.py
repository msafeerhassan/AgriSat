import streamlit as st
import streamlit_authenticator as stauth
import yaml, os
from datetime import date, timedelta
from yaml.loader import SafeLoader
import matplotlib.pyplot as plt
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
from scipy.ndimage import zoom
import numpy as np
from engine import (
    FarmWorkspace,
    # genPolygonRasterMask,
    # genSpectralBand,
    # genCloudMask,
    serializeFarmWorkspace,
    deserializeFarmWorkspace,
    calculateNDVI,
    analyzeZSG,
    genHistoricalRep,
    predictFutureNDVI,
    exportDetailedFarmReport,
    # verifySentinelCredentials,
    downloadAndRegisterSatelliteTelemetry,
    verifyLiveSentinelCredentials,
    excludeAnomolies
)
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

st.set_page_config(page_title="AgriSat Dashboard", page_icon="🛰️", layout="wide")

STORAGE_DIR = "data_store"

if not os.path.exists(STORAGE_DIR):
    os.makedirs(STORAGE_DIR)

with open('auth_config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
)

authenticator.login(location='main')

authenticationStatus = st.session_state.get("authentication_status")
name = st.session_state.get("name")
userName = st.session_state.get("username")

if authenticationStatus is False:
    st.error("Username/Password is incorrect")
    st.stop()
elif authenticationStatus is None:
    st.warning("Please enter your Credentials")
    st.stop()

st.sidebar.title(f"Welcome, {name}!")
st.sidebar.markdown("### Satellite Gateway Status")

@st.cache_data(ttl=300 , show_spinner=False)
def cachedSentinelCheck(clientID: str, clientSecret: str):
    return verifyLiveSentinelCredentials(clientID, clientSecret)

clientID = os.getenv("CLIENT_ID")
clientSecret = os.getenv("CLIENT_SECRET")

connectionValidity, connectionMessage = cachedSentinelCheck(clientID, clientSecret)

if connectionValidity:
    st.sidebar.success(f"SentinelHub API: {connectionMessage}")
else:
    st.sidebar.error(f"SentinelHub API: {connectionMessage}")

st.sidebar.divider()

authenticator.logout("Logout", "sidebar")

actionMode = st.sidebar.radio(
    "Select Action: ",
    [
        "Active Farm Analytics Board",
        "Register & Draw New Farm Boundary"
    ]
)

if actionMode == "Active Farm Analytics Board":
    st.title("Active Farm Analytics Board")
    availableFarms = []
    if os.path.exists(STORAGE_DIR):
        availableFarms = [f.replace("_meta.json", "") for f in os.listdir(STORAGE_DIR) if f.endswith("_meta.json")]
    if not availableFarms:
        st.info("No Farm Registries Deted! Use the 'Register & Draw New Farm Boundary' to generate a Farm Profile first.")
    else:
        selectedFarmID = st.selectbox("Select Target Farm Profile", sorted(availableFarms))

        farm = deserializeFarmWorkspace(selectedFarmID, storageDir=STORAGE_DIR)

        if farm and farm.historicalDates:
            st.subheader(f"Workspace Metrics Dashboard: {farm.farmID} - [{farm.cropType}]")

            anomolyAlerts = excludeAnomolies(farm)
            for alertMsg in anomolyAlerts:
                st.error(f"{alertMsg}")
            
            latestDateObj = sorted(farm.historicalDates)[-1]
            
            latestDateStr = latestDateObj.isoformat()

            ndviMatrix = calculateNDVI(
                farm.redBands[latestDateStr],
                farm.nIRbands[latestDateStr],
                farm.cloudMask[latestDateStr]
            )

            zonalRep = analyzeZSG(ndviMatrix, targetedQuadrant="ALL")
            trendRep = genHistoricalRep(farm)

            mCol1, mCol2, mCol3 = st.columns(3)

            with mCol1:
                st.metric("Latest Snapshot Date:", latestDateStr)
            with mCol2:
                latestCleanMean = float(np.nanmean(ndviMatrix)) if not np.all(np.isnan(ndviMatrix)) else 0.0
                st.metric("Latest Field State Mean NDVI", f"{latestCleanMean:.4f}")
            with mCol3:
                st.metric("Trend Trajectory State", trendRep.get("assessment", "Stable"))

            updateCol1, updateCol2 = st.columns([3, 1])
            with updateCol2:
                updateNowClicked = st.button("Update Now", width='stretch')
            if updateNowClicked:
                with st.spinner(f"Checking for new satellite imagery for {farm.farmID}..."):
                    clientID = os.getenv("CLIENT_ID")
                    clientSecret = os.getenv("CLIENT_SECRET")

                    updateResult = downloadAndRegisterSatelliteTelemetry(farm, clientID, clientSecret)

                    newDatesFetched = sum(
                        1 for entry in updateResult["perDate"] if entry["status"] == "fetched"
                    )
                    if newDatesFetched > 0:
                        serializeFarmWorkspace(farm, storageDir=STORAGE_DIR)
                        st.success(f"Added {newDatesFetched} new snapshots for {farm.farmID}")
                    else:
                        st.info("No new Satellite Data Available!")

                    st.markdown("#### Update Summary")

                    statusLabels = {
                        "fetched": "Successfully Fetched",
                        "too_cloudy": "Too Cloudy",
                        "no_data": "No Data Available",
                        "error": "Error :(",
                        "already_have": "Already Have."
                    }

                    summaryRows = []

                    for entry in updateResult["perDate"]:
                        row = {
                            "Date": entry["date"],
                            "Status": statusLabels.get(entry["status"], entry["status"]),
                        }

                        if entry["status"] == "fetched":
                            row["Mean NDVI"] = f"{entry["meanNDVI"]:.4f}"
                            row["Cloud %"] = f"{entry["cloudPct"]:.1%}"
                        elif entry["status"] == "too_cloudy":
                            row["Cloud %"] = f"{entry["cloudPct"]:.1%}"
                        elif entry["status"] == "error":
                            row["Detail"] = entry["message"]
                        summaryRows.append(row)
                    st.dataframe(summaryRows, width='stretch')
                if newDatesFetched > 0:
                    st.rerun()
            st.divider()

            displayCol, chartCol = st.columns([1, 1])

            with displayCol:
                st.markdown("#### Dynamic Interactive Crop Canopy Map")

                lats = [pt[1] for pt in farm.polygonCoords]
                lons = [pt[0] for pt in farm.polygonCoords]
                centerLat = sum(lats) / len(lats)
                centerLon = sum(lons) / len(lons)

                mapKey = f"analytics_map_{farm.farmID}_{latestDateStr}_v3"

                tileMap = folium.Map(location=[
                    centerLat,
                    centerLon
                ],
                zoom_start=15,
                tiles=None,
                control_scale=True
                )

                folium.TileLayer(
                    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                    attr= 'Tiles &copy; Esri',
                    name="Satellite Imagery",
                    overlay=False,
                    control=True
                ).add_to(tileMap)
                foliumPolyCoords = [[pt[1], pt[0]] for pt in farm.polygonCoords]

                folium.Polygon(
                    locations=foliumPolyCoords,
                    color="#1B5E20",
                    weight=3,
                    fill=False,
                    popup=f"Farm ID: {farm.farmID}"
                ).add_to(tileMap)

                try:

                    cleanNdvi = np.nan_to_num(ndviMatrix, nan=-0.1)

                    upscaledNDvi = zoom(cleanNdvi, zoom=10, order=1)
                
                    rgbaMatrix = np.zeros((upscaledNDvi.shape[0], upscaledNDvi.shape[1], 4), dtype=np.uint8)
                    rgbaMatrix[..., 0] = 34
                    rgbaMatrix[..., 1] = np.clip((upscaledNDvi * 255).astype(int), 0, 255)
                    rgbaMatrix[..., 2] = 34
                    rgbaMatrix[..., 3] = np.where(upscaledNDvi > 0, 191, 0)
                
                    minLon , minLat, maxLon, maxLat = farm.geoBoundary

                    folium.raster_layers.ImageOverlay(
                        image=rgbaMatrix,
                        bounds=[[minLat, minLon], [maxLat, maxLon]],
                        opacity=0.75,
                        interactive=True,
                        z_index = 1
                    ).add_to(tileMap)
                except Exception as overlayErr:
                    st.warning(f"Could Not render NDVI Overlay: {overlayErr}")
                st_folium(
                    tileMap,
                    width=550,
                    height=380,
                    key=mapKey,
                    returned_objects=[]
                )

                st.caption(f"Map Center: {centerLat:.4f}, {centerLon:.4f} | Bounds: {farm.geoBoundary}")
            with chartCol:
                st.markdown("#### Crop Trend Forecasting Charts (Next 3 Cycles)")

                historicalMeans = trendRep["raw_means"]
                predictions = predictFutureNDVI(historicalMeans, projectionSteps=3)

                fig, ax = plt.subplots(figsize = (6, 4.2))
                historyLen = len(historicalMeans)

                ax.plot(range(1, historyLen + 1), historicalMeans, marker="o", color="#2E7D32", linewidth=2, label="Historical Clean Mean")
                ax.plot(range(historyLen, historyLen + 4), [historicalMeans[-1]] + predictions, marker="x", linestyle="--", color="#E65100", linewidth=2, label="Predicted Cycles")

                ax.set_title("Canopy Density Index Trajectory Analysis", fontsize = 10, fontweight="bold")
                ax.set_xlabel("Observation Historical Intervals", fontsize=9)
                ax.set_ylabel("Mean NDVI Matrix Scale", fontsize=9)
                ax.grid(True, alpha=0.3)

                st.pyplot(fig)
            
            st.divider()

            st.markdown("#### Spatial Zonal Diagnosis Breakdown")
            zCol1, zCol2, zCol3, zCol4 = st.columns(4)

            quads = ["NW", "NE", "SW", "SE"]
            cols = [zCol1, zCol2, zCol3, zCol4]

            for q,c in zip(quads, cols):
                qData = zonalRep.get(q, {})
                qMean = qData.get("mean_ndvi", 0.0)
                qDensity = qData.get("coverage_pct", 0.0)
                qCond = qData.get("status", "Unknown")
                with c:
                    st.info(f"Sector {q}")
                    st.markdown(f"Mean NDVI: `{qMean:.4f}`")
                    st.markdown(f"Density Vector: `{qDensity:.1f}%`")
                    st.markdown(f"Condition: {qCond}")
                
            st.divider()

            st.markdown("#### Download Analysis Report")
            savedReportBase = exportDetailedFarmReport(farm, trendRep, zonalRep)
            reportTxtFile = f"{savedReportBase}.txt"

            if os.path.exists(reportTxtFile):
                with open(reportTxtFile, "r") as file:
                    textPayload = file.read()
                
                st.download_button(
                    label="Download Report (.txt)",
                    data=textPayload,
                    file_name=f"{farm.farmID}_Report.txt",
                    mime="text/plain"
                )
        else:
            st.warning("No Assests Found!")

elif actionMode == "Register & Draw New Farm Boundary":
    st.title("Custom Farm Sketchpad")
    st.markdown("Use the Polygon Drawing tool on the left edge of map to trace your Farm's Boundaries.")

    col1, col2 = st.columns(2)
    with col1:
        farmIDInput = st.text_input("Enter Unique Farm ID Identifier like FARM-0001:", "").strip()
    with col2:
        cropTypeInput = st.text_input("Enter the current Crop (Rice, Maize, Potatoes etc): ", "").strip()

    mapCanvas = folium.Map(location=[30.3450, 73.3550], zoom_start=14, tiles=None)

    drawPlugin = Draw(
        draw_options={
            'polyline': False,
            'rectangle': True,
            'polygon': True,
            'circle': False,
            'marker': False,
            'circlemarker': False
        },
        edit_options={
            'remove': True
        }
    )

    drawPlugin.add_to(mapCanvas)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr= 'Tiles &copy;',
        name="Satellite Imagery",
        overlay=False,
        control=True
    ).add_to(mapCanvas)

    mapCaptureData = st_folium(mapCanvas, width=900, height=500, key="sketchpad_drawing_surface")

    if mapCaptureData and "last_active_drawing" in mapCaptureData:
        drawingMeta = mapCaptureData["last_active_drawing"]
        if drawingMeta and "geometry" in drawingMeta:
            geometryType = drawingMeta['geometry']['type']
            if geometryType == "Polygon":
                rawVertices = drawingMeta['geometry']['coordinates'][0]
                formattedCoords = [
                    (float(pt[0]), float(pt[1])) for pt in rawVertices
                ]

                st.success("Geometry Path Vector Extracted Successfully!")

                if st.button("Commit and Register Farm to Workspace"):
                    if not farmIDInput or not cropTypeInput:
                        st.error("Validation Rejected. Please enter Farm ID and Crop Type.")
                    else:
                        with st.spinner("Fetching real telemetry data from SentinelHub API..."):
                            longitudes = [pt[0] for pt in formattedCoords]
                            latitudes = [pt[1] for pt in formattedCoords]
                            computedBbox = (min(longitudes), min(latitudes), max(longitudes), max(latitudes))

                            newFarm = FarmWorkspace(
                                farmID=farmIDInput,
                                cropType=cropTypeInput,
                                geoBoundary=computedBbox,
                                polygonCoords=formattedCoords
                            )

                            clientID = os.getenv("CLIENT_ID")
                            clientSecret = os.getenv("CLIENT_SECRET")

                            downloadResult = downloadAndRegisterSatelliteTelemetry(newFarm, clientID, clientSecret)

                            if downloadResult["success"]:
                                serializeFarmWorkspace(newFarm, storageDir=STORAGE_DIR)
                                st.success(f"Workspace {farmIDInput} successfully Linked with Active Satellite Feed!")
                                st.balloons()
                            else:
                                st.error("Failed to Fetch Satellite Data :(")
                            
                            st.markdown("#### Satellite Fetch Summary")
                            st.caption(
                                f"{downloadResult['successCount']}/{downloadResult['totalAttempted']} dates fetched successfully."
                            )

                            statusLabels = {
                                "fetched": "Fetched :)",
                                "too_cloudy": "Too Cloudy :(",
                                "no_data": "NO DATA :(",
                                "error": "Error :(",
                                "already_have": "Already Have"
                            }

                            summaryRows = []
                            for entry in downloadResult["perDate"]:
                                row = {
                                    "Date": entry["date"],
                                    "Status": statusLabels.get(entry["status"], entry["status"]),
                                }

                                if entry["status"] == "fetched":
                                    row["Mean NDVI"] = f"{entry['meanNDVI']:.4f}"
                                    row["Cloud %"] = f"{entry['cloudPct']:.1%}"
                                elif entry["status"] == "too_cloudy":
                                    row["Cloud %"] = f"{entry['cloudPct']:.1%}"
                                elif entry["status"] == "error":
                                    row["Detail"] = entry["message"]
                                summaryRows.append(row)
                            st.dataframe(summaryRows, width='stretch')