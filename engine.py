from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import numpy as np
from datetime import date
import os, json

@dataclass
class FarmWorkspace:
    farmID: str
    cropType: str
    geoBoundary: Tuple[float, float, float, float]
    historicalDates: List[date] = field(default_factory=list)

    redBands: Dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    nIRbands: Dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    cloudMask: Dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    def addTelemetrySnapshot(self, snapShotDate: date, redBands: np.ndarray, nIRBands: np.ndarray, cloudMask: np.ndarray):
        dateStr = snapShotDate.isoformat()
        if snapShotDate not in self.historicalDates:
            self.historicalDates.append(snapShotDate)
            self.historicalDates.sort()

        self.redBands[dateStr] = redBands
        self.nIRbands[dateStr] = nIRBands
        self.cloudMask[dateStr] = cloudMask

def genSpectralBand(condition: str, bandType: str, shape: Tuple[int, int] = (10, 10)) -> np.ndarray:
    if condition.lower() == "healthy":
        redLow, redHigh = 0.05, 0.15
        nIRLow, nIRHigh = 0.60, 0.85
    elif condition.lower() == "stressed":
        redLow, redHigh = 0.25, 0.35
        nIRLow, nIRHigh = 0.35, 0.45
    else:
        raise ValueError("Condition must be either HEALTHY or STRESSED.")
    
    if bandType.lower() == "red":
        return np.random.uniform(redLow, redHigh, size=shape)
    elif bandType.lower() == "nir":
        return np.random.uniform(nIRLow, nIRHigh, size=shape)
    else: 
        raise ValueError("Band Type must be either RED or NIR(Near Infra-red)")
    
def genCloudMask(shape: Tuple[int, int] = (10, 10), coverageProb: float = 0.2) -> np.ndarray:
    return np.random.rand(*shape) < coverageProb


def calculateNDVI(redBand: np.ndarray, nIRBand: np.ndarray, cloudMask: np.ndarray) -> np.ndarray:
    ndviMatrix = np.full(redBand.shape, np.nan, dtype=float)
    validPixels = ~cloudMask
    redClear = redBand[validPixels]
    nIRClear = nIRBand[validPixels]
    denominators = nIRClear + redClear

    zeroDivisonMask = denominators == 0.0

    computedNDVI = np.zeros_like(denominators)

    safePixels = ~zeroDivisonMask
    computedNDVI[safePixels] = (nIRClear[safePixels] - redClear[safePixels]) / denominators[safePixels]
    computedNDVI[zeroDivisonMask] = 0.0

    ndviMatrix[validPixels] = computedNDVI
    return ndviMatrix


def renderGridMask(ndviMatrix: np.ndarray) -> List[str]:
    renderedRows = []
    for row in ndviMatrix:
        rowChars = []
        for pixel in row:
            if np.isnan(pixel):
                rowChars.append("☁️")
            elif pixel > 0.6:
                rowChars.append("H")
            elif pixel < 0.2:
                rowChars.append(".")
            else:
                rowChars.append("m")
        renderedRows.append(" ".join(rowChars))
    
    return renderedRows

def serializeFarmWorkspace(farm: FarmWorkspace, storageDir: str = "data_store"):
    if not os.path.exists(storageDir):
        os.makedirs(storageDir)

    metaData = {
        "farmID": farm.farmID,
        "cropType": farm.cropType,
        "geoBoundary": farm.geoBoundary,
        "historicalDates": [d.isoformat() for d in farm.historicalDates]
    }

    metaPath = os.path.join(storageDir, f"{farm.farmID}_meta.json")
    with open(metaPath, "w") as f:
        json.dump(metaData, f, indent=4)

    arrayPath = os.path.join(storageDir, f"{farm.farmID}_arrays.npz")
    np.savez_compressed(
        arrayPath,
        redBands=farm.redBands,
        nIRBands=farm.nIRbands,
        cloudMasks = farm.cloudMask
    )

def temporalTLSweeper(farm: FarmWorkspace) -> Dict[str, float]:
    temporalMeans = {}

    for dateStr in farm.redBands.keys():
        red = farm.redBands[dateStr]
        nir = farm.nIRbands[dateStr]
        mask = farm.cloudMask[dateStr]

        ndvi = calculateNDVI(red, nir, mask)

        validNDVI = ndvi[~np.isnan(ndvi)]
        if validNDVI.size > 0:
            temporalMeans[dateStr] = float(np.mean(validNDVI))
        else:
            temporalMeans[dateStr] = 0.0
    
    return temporalMeans

def excludeAnomolies(farm: FarmWorkspace) -> List[str]:
    alerts = []
    meansTimeline = temporalTLSweeper(farm)
    sortedDates = sorted(meansTimeline.keys())

    for i in range(1, len(sortedDates)):
        prevDate = sortedDates[i-1]
        currDate = sortedDates[i]

        prevMean = max(0.0, meansTimeline[prevDate])
        currMean = max(0.0, meansTimeline[currDate])

        if prevMean > 0:
            dropPercentage = (prevMean - currMean) / prevMean
            if dropPercentage > 0.20:
                alerts.append(f"Anomoly Detected at Farm: {farm.farmID}. Health Dropped by {dropPercentage * 100}% since {prevDate} till {currDate}")
    
    return alerts