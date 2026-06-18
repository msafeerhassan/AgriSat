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
    genPolygonRasterMask,
    genSpectralBand,
    genCloudMask,
    serializeFarmWorkspace,
    deserializeFarmWorkspace,
    calculateNDVI,
    analyzeZSG,
    genHistoricalRep,
    predictFutureNDVI,
    exportDetailedFarmReport
)

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
                st.metric("Trend Trajectory State", trendRep.get("trajectory", "Stable"))

            st.divider()

            displayCol, chartCol = st.columns([1, 1])

            with displayCol:
                st.markdown("#### Dynamic Interactive Crop Canopy Map")

                lats = [pt[1] for pt in farm.polygonCoords]
                lons = [pt[0] for pt in farm.polygonCoords]
                centerLat = sum(lats) / len(lats)
                centerLon = sum(lons) / len(lons)

                tileMap = folium.Map(location=[
                    centerLat,
                    centerLon
                ],
                zoom_start=15,
                tiles="OpenStreetMap"
                )

                foliumPolyCoords = [[pt[1], pt[0]] for pt in farm.polygonCoords]

                folium.Polygon(
                    locations=foliumPolyCoords,
                    color="#1B5E20",
                    weight=3,
                    fill=False,
                    popup=f"Farm ID: {farm.farmID}"
                ).add_to(tileMap)

                cleanNdvi = np.nan_to_num(ndviMatrix, nan=-0.1)

                upscaledNDvi = zoom(cleanNdvi, zoom=10, order=1)

                def colorAlphaScale(val):
                    if val <= 0:
                        return (0, 0, 0, 0)
                    greenIntensity = int(max(0, min(255, val * 255)))
                    return (34, greenIntensity, 34, int(0.75 * 255))
                
                rgbaMatrix = np.zeros((upscaledNDvi.shape[0], upscaledNDvi.shape[1], 4), dtype=np.uint8)
                for r in range(upscaledNDvi.shape[0]):
                    for c in range(upscaledNDvi.shape[1]):
                        rgbaMatrix[r, c] = colorAlphaScale(upscaledNDvi[r, c])
                
                minLon , minLat, maxLon, maxLat = farm.geoBoundary

                folium.raster_layers.ImageOverlay(
                    image=rgbaMatrix,
                    bounds=[[minLat, minLon], [maxLat, maxLon]],
                    opacity=0.8
                ).add_to(tileMap)

                st_folium(tileMap, width=550, height=380, key=f"analytics_map_viewer_{farm.farmID}")
            with chartCol:
                st.markdown("#### Crop Trend Forecasting Charts (Next 3 Cycles)")

                sortedDates = sorted(farm.historicalDates)
                historicalMeans = []
                for d in sortedDates:
                    dStr = d.isoformat()
                    mNDVI = calculateNDVI(farm.redBands[dStr], farm.nIRbands[dStr], farm.cloudMask[dStr])
                    historicalMeans.append(float(np.nanmean(mNDVI)) if not np.all(np.isnan(mNDVI)) else 0.0)

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

    mapCanvas = folium.Map(location=[30.3450, 73.3550], zoom_start=14, tiles="OpenStreetMap")

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
                        with st.spinner("Processing Data..."):
                            longitudes = [pt[0] for pt in formattedCoords]
                            latitudes = [pt[1] for pt in formattedCoords]
                            computedBbox = (min(longitudes), min(latitudes), max(longitudes), max(latitudes))

                            newFarm = FarmWorkspace(
                                farmID=farmIDInput,
                                cropType=cropTypeInput,
                                geoBoundary=computedBbox,
                                polygonCoords=formattedCoords
                            )

                            gridShape = (15, 15)

                            polyMask = genPolygonRasterMask(newFarm, gridShape)

                            for lookback in [20, 15, 10, 5]:
                                snapDate = date.today() - timedelta(days=lookback)

                                rBand = genSpectralBand("healthy", "red", gridShape)
                                nirBand = genSpectralBand("healthy", "nir", gridShape)
                                cMask = genCloudMask(gridShape, coverageProb=0.0)

                                rBand[~polyMask] = np.nan
                                nirBand[~polyMask] = np.nan
                                cMask[~polyMask] = True

                                newFarm.addTelemetrySnapshot(snapDate, rBand, nirBand, cMask)
                            
                            serializeFarmWorkspace(newFarm, storageDir=STORAGE_DIR)
                            st.success(f"Farm {farmIDInput} saved successfully!")
                            st.balloons()
