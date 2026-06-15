from datetime import date
import numpy as np
from engine import FarmWorkspace, genCloudMask, genSpectralBand

def dummyDataTests():
    print("AgriSat Engine Testing")
    print("-" * 30)

    testFarm = FarmWorkspace(
        farmID="FARM-0001",
        cropType="Rice",
        geoBoundary=(34.0522, -118.2437, 34.0622, -118.2337)
    )

    assert testFarm.farmID == "FARM-0001", "Fail: Farm ID isn't correct"
    assert testFarm.cropType == "Rice", "Fail: Crop Type is wrong"
    # print("Farm Object Initialized with correct strucutre")

    testShape = (10, 10)
    redSpectMatrix = genSpectralBand("healthy", "red", shape=testShape)

    assert isinstance(redSpectMatrix, np.ndarray), "Failed: Output isn't NumPy array"
    assert redSpectMatrix.shape == testShape , f"Output shape is {redSpectMatrix} instead of {testShape}"

    healthyRed = genSpectralBand("healthy", "red")
    healthyNIR = genSpectralBand("healthy", "nir")

    stressedRed = genSpectralBand("stressed", "red")
    stressedNIR = genSpectralBand("stressed", "nir")

    assert np.all((healthyRed >= 0.05) & (healthyRed <= 0.15)), "Failed: Healthy Red out of the expected bounds"
    assert np.all((healthyNIR >= 0.60) & (healthyNIR <= 0.85)), "Failed: Healthy NIR out the expectd bounds"
    assert np.all((stressedRed >= 0.60) & (stressedRed <= 0.85)), "Failed: Stressed Red out of expected boundary"
    assert np.all((stressedNIR >= 0.05) & (stressedNIR <= 0.15)), "Failed: Stressed NIR out of the expected bounds."

    cloudyMask = genCloudMask(shape=testShape, coverageProb=0.3)

    assert cloudyMask.dtype == bool, "Failed: It must be bool"
    assert cloudyMask.shape == testShape, "Failed: Cloud mask shape mismatched the expected"

    evaluationDate = date(2026, 6, 15)
    testFarm.addTelemetrySnapshot(
        snapShotDate=evaluationDate,
        redBands=healthyRed,
        nIRBands=healthyNIR,
        cloudMask=cloudyMask
    )

    dateKey = evaluationDate.isoformat()

    assert dateKey in testFarm.redBands, "Failed: Red Band save failed"
    assert dateKey in testFarm.cloudMask, "Cloud mask save failed"

    print("All tests passed ig")
dummyDataTests()