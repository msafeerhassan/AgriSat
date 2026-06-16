import sys
from datetime import date, timedelta
import numpy as np
from engine import FarmWorkspace, genCloudMask, genSpectralBand, calculateNDVI, renderGridMask, temporalTLSweeper, excludeAnomolies, serializeFarmWorkspace, exportNDVIHeatMap, deserializeFarmWorkspace
from utils import displayTabSummary

# def dummyInitialDataTests():
#     print("AgriSat Engine Testing")
#     print("-" * 30)

#     testFarm = FarmWorkspace(
#         farmID="FARM-0001",
#         cropType="Rice",
#         geoBoundary=(34.0522, -118.2437, 34.0622, -118.2337)
#     )

#     assert testFarm.farmID == "FARM-0001", "Fail: Farm ID isn't correct"
#     assert testFarm.cropType == "Rice", "Fail: Crop Type is wrong"
#     # print("Farm Object Initialized with correct strucutre")

#     testShape = (10, 10)
#     redSpectMatrix = genSpectralBand("healthy", "red", shape=testShape)

#     assert isinstance(redSpectMatrix, np.ndarray), "Failed: Output isn't NumPy array"
#     assert redSpectMatrix.shape == testShape , f"Output shape is {redSpectMatrix} instead of {testShape}"

#     healthyRed = genSpectralBand("healthy", "red")
#     healthyNIR = genSpectralBand("healthy", "nir")

#     stressedRed = genSpectralBand("stressed", "red")
#     stressedNIR = genSpectralBand("stressed", "nir")

#     assert np.all((healthyRed >= 0.05) & (healthyRed <= 0.15)), "Failed: Healthy Red out of the expected bounds"
#     assert np.all((healthyNIR >= 0.60) & (healthyNIR <= 0.85)), "Failed: Healthy NIR out the expectd bounds"
#     assert np.all((stressedRed >= 0.60) & (stressedRed <= 0.85)), "Failed: Stressed Red out of expected boundary"
#     assert np.all((stressedNIR >= 0.05) & (stressedNIR <= 0.15)), "Failed: Stressed NIR out of the expected bounds."

#     cloudyMask = genCloudMask(shape=testShape, coverageProb=0.3)

#     assert cloudyMask.dtype == bool, "Failed: It must be bool"
#     assert cloudyMask.shape == testShape, "Failed: Cloud mask shape mismatched the expected"

#     evaluationDate = date(2026, 6, 15)
#     testFarm.addTelemetrySnapshot(
#         snapShotDate=evaluationDate,
#         redBands=healthyRed,
#         nIRBands=healthyNIR,
#         cloudMask=cloudyMask
#     )

#     dateKey = evaluationDate.isoformat()

#     assert dateKey in testFarm.redBands, "Failed: Red Band save failed"
#     assert dateKey in testFarm.cloudMask, "Cloud mask save failed"

#     # print("All tests passed ig")

#     ndviResult = calculateNDVI(healthyRed, healthyNIR, cloudyMask)

#     assert ndviResult.shape == testShape, "Failed: NDVI Matrix mismatch in the shape"
#     nanLocations = np.isnan(ndviResult)

#     assert np.array_equal(nanLocations, cloudyMask), "Failed: Cloud masks not mapped to NaN entries"

#     clearHealthyValues = ndviResult[~cloudyMask]
#     if clearHealthyValues.size > 0:
#         assert np.all(clearHealthyValues > 0.5), "Failed: Healthy crop yeild low NDVi score"

#     print("Terminal Grid Preview")
#     textGrid = renderGridMask(ndviResult)
#     for rowString in textGrid:
#         print(rowString)
    
# def temporalSimTest():
#     testFarm = FarmWorkspace(
#         farmID="FARM-0001",
#         cropType="Rice",
#         geoBoundary=(34.0522, -118.2437, 34.0622, -118.2337)
#     )
#     baseDate = date(2026, 5, 1)

#     for week in range(5):
#         snapShotDate = baseDate + timedelta(weeks=week)

#         if week == 4:
#             red = genSpectralBand("stressed", "red")
#             nir = genSpectralBand("stressed", "nir")
#         else: 
#             red = genSpectralBand("healthy", "red")
#             nir = genSpectralBand("healthy", "nir")
        
#         mask = genCloudMask(coverageProb=0.15)
#         testFarm.addTelemetrySnapshot(snapShotDate, red, nir, mask)

#     serializeFarmWorkspace(testFarm)
#     print("data stored")

#     weeklyMeans = temporalTLSweeper(testFarm)
#     for dateStr, meanValue in weeklyMeans.items():
#         print(f"Week {dateStr}: Mean NDVI: {meanValue:.4f}")
    
#     activeAlarms = excludeAnomolies(testFarm)

#     for alarm in activeAlarms:
#         print(alarm)


    
#     assert len(activeAlarms) > 0, "Error: Engine didn't flagged suspected issues"


# dummyInitialDataTests()

# temporalSimTest()

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
        print("[3] Generate Anomolous Stress Alerts")
        print("[4]: Exit")
        print("-" * 50)
        try:
            userInput = int(input("Select which action to perform: "))

            if userInput != 1 and userInput != 2 and userInput != 3 and userInput != 4:
                print("Please either choose 1, 2, 3 or 4!")
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
            print("Exiting :(")
            sys.exit(0)
        else:
            print("Unidentified Input!")        
    
runInteractiveDashboard()