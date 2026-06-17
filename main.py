import sys, os
from datetime import date, timedelta
import numpy as np
from dotenv import load_dotenv
from engine import FarmWorkspace, genCloudMask, genSpectralBand, calculateNDVI, renderGridMask, temporalTLSweeper, excludeAnomolies, serializeFarmWorkspace, exportNDVIHeatMap, deserializeFarmWorkspace, analyzeZSG, genHistoricalRep, exportDetailedFarmReport, verifySentinelCredentials, genSentinelNDVIReq, verifyAPIConnectionMock, verifyMatrixReshaping, downloadAndRegisterSatelliteTelemetry, smoothenTemporalNDVI, predictFutureNDVI
from utils import displayTabSummary

load_dotenv()

print(os.getenv("CLIENT_ID"))

def seedInitWorkspace() -> dict:
    registry = {}
    targetIDs = ["FARM-0001", "FARM-0002", "FARM-0003"]
    loadedFromDisk = False

    for fID in targetIDs:
        loadedFarm = deserializeFarmWorkspace(fID)
        if loadedFarm:
            registry[fID] = loadedFarm
            loadedFromDisk = True

    if loadedFromDisk:
        print("Existing Farm Profiles Identified and Restored")
        return registry
    
    print("No saved record found. Using Dummy Data")

    cropTypes = ["Rice", "Wheat", "Cotton"]

    for i, crop in enumerate(cropTypes, 1):
        fID = f"FARM-000{i}"
        farm = FarmWorkspace(
            farmID=fID,
            cropType=crop,
            geoBoundary=(34+i*0.01, -118.24, 34.06 + i*0.01, -118.23)
        )

        baseDate = date(2026, 5, 1)

        for week in range(5):
            snapDate = baseDate + timedelta(weeks=week)
            if fID == "FARM-0001" and week == 4:
                red = genSpectralBand("stressed", "red")
                nir = genSpectralBand("stressed", "nir")
            else:
                red = genSpectralBand("healthy", "red")
                nir = genSpectralBand("healthy", "nir")
            
            mask = genCloudMask(coverageProb=0.1)
            farm.addTelemetrySnapshot(snapDate, red, nir, mask)
            serializeFarmWorkspace(farm)
            registry[fID] = farm
    return registry

def runInteractiveDashboard():
    farmDb = seedInitWorkspace()

    while True:
        print("\n" + "-" * 50)
        print("          AGRISAT CONTROL ENGINE          ")
        print("-" * 50)
        print("[1]: View Registered Farms")
        print("[2]: Run Satellite Diagnostic Scan")
        print("[3]: Generate Anomolous Stress Alerts")
        print("[4]: Upload Real-Time Satellite Telemetry")
        print("[5]: Perform Spatial Zonal Diagnostics")
        print("[6]: Analyze Historical Trends")
        print("[7]: Export Analytics Reports")
        print("[8]: Exit :(")
        print("-" * 50)
        try:
            userInput = int(input("Select which action to perform: "))

            if userInput != 1 and userInput != 2 and userInput != 3 and userInput != 4 and userInput != 5 and userInput != 6 and userInput != 7 and userInput != 8:
                print("Please either choose 1, 2, 3, 4, 5, 6, 7 or 8!")
                continue
        except ValueError:
            print("Invalid Entry.")
            continue
        
        if userInput == 1:
            verifyMatrixReshaping()

            verifySentinelCredentials()

            print("\nRunning Geospatial Validation Test")
            uiList = []
            for fID, fObj in farmDb.items():
                try:
                    spatialMeta = fObj.getValidatedBounds()
                    print(f"Valid {fID}: {spatialMeta['crs']} | {spatialMeta['area_deg_sq']:.6f} sq deg.")
                    verifyAPIConnectionMock(fObj)
                except Exception as geoErr:
                    print(f"Error: {geoErr}")
                uiList.append({
                    "id": fObj.farmID,
                    "crop": fObj.cropType,
                    "bounds": f"{fObj.geoBoundary[0]}, {fObj.geoBoundary[1]}"
                })
            
            displayTabSummary(uiList)
        elif userInput == 2:
            print("\nSelect target farm registry workspace: ")
            availableIDs = list(farmDb.keys())
            for idx, fId in enumerate(availableIDs, 1):
                print(f"[{idx}]: {fId}")
            
            try:
                userConsent = int(input("Enter choice number index: ")) - 1
                if not (0 <= userConsent < len(availableIDs)):
                    print("Out of range :(")
                    continue
                targetID = availableIDs[userConsent]
                farm = farmDb[targetID]

                if not farm.historicalDates:
                    print("No telemtry data available!")
                    continue

                latestDateStr = sorted(farm.redBands.keys())[-1]
                redArr = farm.redBands[latestDateStr]
                nirArr = farm.nIRbands[latestDateStr]
                maskArr = farm.cloudMask[latestDateStr]

                ndviMatrix = calculateNDVI(redArr, nirArr, maskArr)

                validValues = ndviMatrix[~np.isnan(ndviMatrix)]
                meanNdvi = float(np.mean(validValues)) if validValues.size > 0 else 0.0

                print(f"Calculated Farm Mean NDVI: {meanNdvi:.4f}")

                print("Generating 2D Color-mapped Plot Asset")

                savedPlotFile = exportNDVIHeatMap(farm.farmID, latestDateStr, ndviMatrix)

                print("-" * 50)
                print("           Diagnostic Scan and Rendering Complete!")
                print("-" * 50)
                if savedPlotFile:
                    print(f"Exported at: {savedPlotFile}")
                print(f"Overall Field State Mean: {meanNdvi:.4f}")
                print("-" * 50)
            except ValueError:
                print("Invaid Entry Sequence Parameter")  
        elif userInput == 3:
            print("\nEvaluating Farm Profiles in Database and detecting anomolies...")
            alertFound = False
            for farmObj in farmDb.values():
                activeAlerts = excludeAnomolies(farmObj)
                for alert in activeAlerts:
                    print(alert)
                    alertFound = True
            if not alertFound:
                print("Everything is OK uptil now :)")     
        elif userInput == 4:
            print("\nSelect target farm profile to append the telemtry Data: ")
            availableIDs = list(farmDb.keys())
            for idx, fId in enumerate(availableIDs, 1):
                print(f"[{idx}: {fId}]")
            
            try:
                farmChoice = int(input("Enter the farm index to continue with: ")) - 1
                if not (0 <= farmChoice < len(availableIDs)):
                    print("Out of Range :)")
                    continue
                targetID = availableIDs[farmChoice]
                selectedFarm = farmDb[targetID]

                print(f"\nEnter the date for new snapshots insertion: ")
                year = int(input("Enter year: "))
                month = int(input("Enter Month: "))
                day = int(input("Enter day"))
                inputDate = date(year, month, day)

                if inputDate.isoformat() in selectedFarm.redBands:
                    print(f"Matrix Data already present for {inputDate.isoformat()}.")
                    continue

                print("\nChoose Telemetry Data Intake Source: ")
                print("[1]: Download Live Sentinel-2 Satellite Telemetry")
                print("[2]: Use Mock Simulation")
                dataSourceChoice = int(input("Select 1 or 2: "))

                if dataSourceChoice == 1:
                    success = downloadAndRegisterSatelliteTelemetry(selectedFarm, inputDate)
                    if not success:
                        print("Failed fetching satellite data.")
                        continue
                else:
                    print("\nSelect crop health status: ")
                    print("[1]: Optimally Healthy Vegetation")
                    print("[2]: Environmental Stress")
                    condChoice = int(input("Select 1 or 2:"))

                    conditionStr = "healthy" if condChoice == 1 else "stressed"

                    redMatrix = genSpectralBand(conditionStr, "red")
                    nirMatrix = genSpectralBand(conditionStr, "nir")
                    cloudMatrix = genCloudMask(coverageProb=0.12)

                    selectedFarm.addTelemetrySnapshot(inputDate, redMatrix, nirMatrix, cloudMatrix)

                    serializeFarmWorkspace(selectedFarm)
                    print(f"Telemetry Snapshot for {inputDate.isoformat()} successfully added")
            except ValueError:
                print("Invalid Entry!")
        elif userInput == 5:
            print("\nSelect target farm: ")
            availableIDs = list(farmDb.keys())
            for idx, Fid in enumerate(availableIDs, 1):
                print(f"[{idx}: {Fid}]")
            
            try:
                farmChoice = int(input("Enter the farm index to continue with: ")) - 1
                if not (0 <= farmChoice < len(availableIDs)):
                    print("Out of Range :)")
                    continue
                targetID = availableIDs[farmChoice]
                farm = farmDb[targetID]

                if not farm.historicalDates:
                    print("No Telemetry Data found for this profie")
                    continue
                latestDateStr = sorted(farm.redBands.keys())[-1]
                ndviMatrix = calculateNDVI(farm.redBands[latestDateStr], farm.nIRbands[latestDateStr], farm.cloudMask[latestDateStr])

                fullReport = analyzeZSG(ndviMatrix, targetedQuadrant="ALL")

                print(f"\n" + "-" * 50)
                print(f"Spatial Zonal Diagnostics: {targetID}")
                print(f"Target Snapshot Date: {latestDateStr}")
                print("-" * 50)

                for quad, data in fullReport.items():
                    print(f"Sector [{quad}]: Mean NDVI: {data['mean_ndvi']:.4f} | Density: {data['coverage_pct']}% | Status: {data['status']}")
                print("-" * 50)
            except ValueError:
                print("Format Error")
        elif userInput == 6:
            print("\nSelect target farm profile to append the telemtry Data: ")
            availableIDs = list(farmDb.keys())
            for idx, fId in enumerate(availableIDs, 1):
                print(f"[{idx}: {fId}]")
            
            try:
                farmChoice = int(input("Enter the farm index to continue with: ")) - 1
                if not (0 <= farmChoice < len(availableIDs)):
                    print("Out of Range :)")
                    continue
                targetID = availableIDs[farmChoice]
                farm = farmDb[targetID]

                if not farm.historicalDates or len(farm.redBands) == 0:
                    print("No Telemtry Data Found!")
                    continue
                print("\n" + "-" * 50)
                print(f"Historical Trend Profile & Predictive Insights")
                print("-" * 50)
                report = genHistoricalRep(farm)
                print(f"Farm Identity: {farm.farmID} ({farm.cropType})")
                print(f"Slope: {report['overall_slope']:.6f}")
                print(f"Smoothed Vector: {report['trend_vector']}")
                print(f"Current Status: {report['assessment']}")
                print("-" * 50)

                rawMeans = []
                for d in farm.historicalDates:
                    dateStr = d.isoformat()
                    ndvi = calculateNDVI(farm.redBands[dateStr], farm.nIRbands[dateStr] , farm.cloudMask[dateStr])
                    valid = ndvi[~np.isnan(ndvi)]
                    rawMeans.append(float(np.mean(valid)) if valid.size > 0 else 0.0)
                cleanMeans = smoothenTemporalNDVI(rawMeans, windowSize=3)

                futurePredict = predictFutureNDVI(cleanMeans, projectionSteps=3)

                print(f"Seasonal Expectations Forecasting of Next 3 Cycles")
                for cycle, projectVal in enumerate(futurePredict, 1):
                    if projectVal < 0.25:
                        alertTag = "Risk!"
                    else:
                        alertTag = "Safe!"
                    
                    print(f"Cycle +{cycle} Projection: Expected NDVI: {projectVal} | Status: {alertTag}")
                print("-" * 50)
            except ValueError:
                print("Error")
        elif userInput == 7:
            print("\nSelect target farm profile to append the telemtry Data: ")
            availableIDs = list(farmDb.keys())
            for idx, fId in enumerate(availableIDs, 1):
                print(f"[{idx}: {fId}]")
            
            try:
                farmChoice = int(input("Enter the farm index to continue with: ")) - 1
                if not (0 <= farmChoice < len(availableIDs)):
                    print("Out of Range :(")
                    continue
                targetID = availableIDs[farmChoice]
                farm = farmDb[targetID]

                if not farm.historicalDates:
                    print("No Telemetry Records Found for this Farm.")
                    continue
                trendReport = genHistoricalRep(farm)
                latestDateStr = sorted(farm.redBands.keys())[-1]

                ndviMatrix = calculateNDVI(farm.redBands[latestDateStr], farm.nIRbands[latestDateStr], farm.cloudMask[latestDateStr])

                zonalRep = analyzeZSG(ndviMatrix, targetedQuadrant="ALL")

                savedPathBase = exportDetailedFarmReport(farm, trendReport, zonalRep)

                print(f"\n" + "-" * 50)
                print(f"           Report Exported Successfully!")
                print("-" * 50)
                print(f"Raw Data saved at {savedPathBase}.json")
                print(f"Executive Briefing Saved at: {savedPathBase}.txt")
                print(f"Saved Reports into 'outputReports/' folder")
                print("-" * 50)
            except ValueError:
                print("Error")
        else:
            print("Exiting :(")
            sys.exit(0)
    
runInteractiveDashboard()