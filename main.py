import sys
from datetime import date, timedelta
import numpy as np
from engine import FarmWorkspace, genCloudMask, genSpectralBand, calculateNDVI, renderGridMask, temporalTLSweeper, excludeAnomolies, serializeFarmWorkspace, exportNDVIHeatMap, deserializeFarmWorkspace, analyzeZSG, genHistoricalRep
from utils import displayTabSummary

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
        print("[7]: Exit :(")
        print("-" * 50)
        try:
            userInput = int(input("Select which action to perform: "))

            if userInput != 1 and userInput != 2 and userInput != 3 and userInput != 4 and userInput != 5 and userInput != 6 and userInput != 7:
                print("Please either choose 1, 2, 3, 4, 5, 6 or 7!")
                continue
        except ValueError:
            print("Invalid Entry.")
            continue
        
        if userInput == 1:
            uiList = []
            for fID, fObj in farmDb.items():
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
                if 0 <= userConsent < len(availableIDs):
                    targetID = availableIDs[userConsent]
                    farm = farmDb[targetID]

                    latestDateStr = sorted(farm.redBands.keys())[-1]
                    redArr = farm.redBands[latestDateStr]
                    nirArr = farm.nIRbands[latestDateStr]
                    maskArr = farm.cloudMask[latestDateStr]

                    ndviMatrix = calculateNDVI(redArr, nirArr, maskArr)

                    print(f"\n --- Visual Terminal Grid Map Preview ({targetID} | {latestDateStr}) --- ")
                    textGrid = renderGridMask(ndviMatrix)

                    for rowString in textGrid:
                        print(rowString)
                    
                    exportNDVIHeatMap(ndviMatrix, farm.farmID)
                else:
                    print("Error!")
            except (ValueError, IndexError):
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

                print(f"\nSelect target date: ")
                availableDates = sorted(list(farm.redBands.keys()))
                for idx, dstr in enumerate(availableDates, 1):
                    print(f"[{idx}]: [{dstr}]")

                dateChoice = int(input("Enter choice index: ")) - 1

                if not (0 <= dateChoice < len(availableDates)):
                    print("Out of range")
                    continue
                targetDateStr = availableDates[dateChoice]

                ndviMatrix = calculateNDVI(
                    farm.redBands[targetDateStr],
                    farm.nIRbands[targetDateStr],
                    farm.cloudMask[targetDateStr]
                )

                print("\nSelect target sector location: ")
                print("1. Northwest")
                print("2. Northeast")
                print("3. Southwest")
                print("4. Southeast")
                quadCode = int(input("Enter Sector index: "))

                if quadCode != 1 and quadCode !=2 and quadCode != 3 and quadCode != 4:
                    print("Wrong Selection")
                    continue
                
                if quadCode == 1:
                    quadCode = "NW"
                elif quadCode == 2:
                    quadCode = "NE"
                elif quadCode == 3:
                    quadCode = "SW"
                elif quadCode == 4:
                    quadCode = "SE"
                
                stats = analyzeZSG(ndviMatrix, quadCode)

                print(f"\n" + "-" * 45)
                print(f"Spatial Analysis Report: {targetID} ({quadCode})")
                print(f"Timestmap: {targetDateStr}")
                print("-" * 45)
                print(f"Status: {stats['status']}")
                print(f"Mean NDVI: {stats['mean']:.4f}")
                print(f"Maximum NDVI: {stats['max']:.4f}")
                print(f"Minimum NDVI: {stats['min']:.4f}")
                print(f"Valid Data Coverage: {stats['coverage_pct']:.1f}%")
                print("-" * 45)
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

                report = genHistoricalRep(farm)

                print("\n" + "-" * 50)
                print(f"Historical Trend Profile of: {targetID}")
                print("-" * 50)
                print(f"Crop Variety: {farm.cropType}")
                print(f"Monitoring Date Range: {report['dates'][0]} to {report['dates'][-1]}")
                print(f"Slope: {report['overall_slope']:.4f}")
                print(f"Trend Vector: {report['trend_vector']}")
                print(f"Remarks: {report['assessment']}")
                print("-" * 50)

                print("\nTimeline Breakdown: ")
                for d, m in zip(report['dates'], report['means']):
                    print(f"{d} Average NDVI: {m:.4f}")
                
                print("-" * 50)
            except ValueError:
                print("Error")
        else:
            print("Exiting :(")
            sys.exit(0)
    
runInteractiveDashboard()