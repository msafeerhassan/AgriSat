import streamlit as st
import streamlit_authenticator as stauth
import yaml, os
from datetime import date, timedelta
from yaml.loader import SafeLoader
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
import numpy as np
from engine import (
    FarmWorkspace,
    genPolygonRasterMask,
    genSpectralBand,
    genCloudMask,
    serializeFarmWorkspace
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
    st.info("Coming Soon...")
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
