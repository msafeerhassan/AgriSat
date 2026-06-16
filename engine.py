from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import numpy as np
from datetime import date
import os, json, pickle
import matplotlib.pyplot as plt

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

    arrayPath = os.path.join(storageDir, f"{farm.farmID}_arrays.pkl")

    with open(arrayPath, "wb") as f:
        pickle.dump({
            "redBands": farm.redBands,
            "nirBands": farm.nIRbands,
            "cloudMasks": farm.cloudMask
        }, f)

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

def exportNDVIHeatMap(ndviMatrix: np.ndarray, farmID: str, outputDir: str = "outputReports"):
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    
    plt.figure(figsize=(6, 5))

    currentCMap = plt.cm.YlGn.copy()
    currentCMap.set_bad(color="darkgray")

    img = plt.imshow(ndviMatrix, cmap=currentCMap, vmax=1.0, vmin=0.0)
    plt.colorbar(img, label="NDVI Intesnity Index Value")

    plt.title(f"AgriSat Satellite Performance Mapping: {farmID}")
    plt.xlabel("Grid X-Axis Coordinate")
    plt.ylabel("Grid Y-Axis Coordinate")

    outputPath = os.path.join(outputDir, f"{farmID}HeatMap.png")
    plt.savefig(outputPath, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Graphical Map Generated and Saved at: {outputPath}")

def deserializeFarmWorkspace(farmID: str, storageDir: str = "data_store") -> Optional[FarmWorkspace]:
    metaPath = os.path.join(storageDir, f"{farmID}_meta.json")
    arrayPath = os.path.join(storageDir, f"{farmID}_arrays.pkl")

    if not os.path.exists(metaPath) or not os.path.exists(arrayPath):
        return None
    with open(metaPath, "r") as file:
        metaData = json.load(file)
    
    farm = FarmWorkspace(
        farmID=metaData["farmID"],
        cropType=metaData["cropType"],
        geoBoundary=tuple(metaData["geoBoundary"])
    )

    farm.historicalDates = [date.fromisoformat(d) for d in metaData["historicalDates"]]

    with open(arrayPath, "rb") as f:
        arrays = pickle.load(f)
        farm.redBands = arrays["redBands"]
        farm.nIRbands = arrays["nirBands"]
        farm.cloudMask = arrays["cloudMasks"]
    return farm

def analyzeZSG(ndviMatrix: np.ndarray, quadrant: str) -> dict:
    if quadrant.lower() == "nw":
        rowStart, rowEnd, colStart, colEnd = 0, 5, 0, 5
    elif quadrant.lower() == "ne":
        rowStart, rowEnd, colStart, colEnd = 0, 5, 5, 10
    elif quadrant.lower() == "sw":
        rowStart, rowEnd, colStart, colEnd = 5, 10, 0, 5
    elif quadrant.lower() == "se":
        rowStart, rowEnd, colStart, colEnd = 5, 10, 5, 10
    else: 
        raise ValueError("Unknow orientations.")
    
    zonalSlice = ndviMatrix[rowStart:rowEnd, colStart:colEnd]

    validPix = zonalSlice[~np.isnan(zonalSlice)]

    if validPix.size == 0:
        return {
            "mean": 0.0,
            "max": 0.0,
            "min": 0.0,
            "coverage_pct": 0.0,
            "status": "Obscured (100% Cloud Cover)"
        }
    meanVal = float(np.mean(validPix))
    coveragePct = (validPix.size / zonalSlice.size) * 100

    if meanVal > 0.7:
        status = "Excellent - High Biomass"
    elif meanVal > 0.4:
        status = "Moderate - Stable Development"
    else:
        status = "Critical - Low Vegetation Density"
    
    return {
        "mean": meanVal,
        "max": float(np.max(validPix)),
        "min": float(np.min(validPix)),
        "coverage_pct": coveragePct,
        "status": status
    }

def genHistoricalRep(farm: FarmWorkspace) -> Dict:
    meansTL = temporalTLSweeper(farm)
    sortedDates = sorted(meansTL.keys())

    if len(sortedDates) < 2:
        return {
            "dates": sortedDates,
            "means": [meansTL.get(d, 0,0) for d in sortedDates],
            "overall_slope": 0.0,
            "trend_vector": "Insufficient Data Sequence",
            "assessment": "Need atleast 2 historical frames to evaluate trends"
        }
    meansList = [meansTL[d] for d in sortedDates]

    slopes = []
    for i in range(1, len(meansList)):
        slopes.append(meansList[i] - meansList[i-1])

    overallSlope = meansList[-1] - meansList[0]

    if overallSlope > 0.15:
        assesment = "Growth!"
        trendVector = "📈 Upward Growth (".join([f"{m:.2f}" for m in meansList]) + ")"
    elif overallSlope <-0.15:
        assesment = "Decline!"
        trendVector = "📉 Downward Growth (".join([f"{m:.2f}" for m in meansList]) + ")"
    else:
        assesment = "Stable!"
        trendVector = "Stable (".join([f"{m:.2f}" for m in meansList]) + ")"
    
    return {
        "dates": sortedDates,
        "means": meansList,
        "overall_slope": overallSlope,
        "trend_vector": trendVector,
        "assessment": assesment
    }

def exportDetailedFarmReport(farm: FarmWorkspace, trendReport: dict, zonalReport: dict, quadCode: str, outputDir: str = "outputReports") -> str:
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    basePath = os.path.join(outputDir, f"{farm.farmID}_Summary")
    jsonPayload = {
        "farmID": farm.farmID,
        "cropType": farm.cropType,
        "coordinates": farm.geoBoundary,
        "temporalTrends": {
            "monitored_dates": trendReport["dates"],
            "historical_means": trendReport["means"],
            "seasonal_slope": float(trendReport["overall_slope"]),
            "trajectory_assessment": trendReport["assessment"]
        },
        "spatial_zone_analysis": {
            "analyzed_sector": quadCode,
            "mean_ndvi": zonalReport["mean"],
            "max_ndvi": zonalReport["max"],
            "min_ndvi": zonalReport["min"],
            "data_coverage_pct": zonalReport["coverage_pct"],
            "sector_status": zonalReport["status"]
        },
        "compiled_timestamp": date.today().isoformat()
    }

    with open(f"{basePath}.json", "w") as file:
        json.dump(jsonPayload, file, indent=4)

    with open(f"{basePath}.txt", "w", encoding="utf-8") as tf:
        tf.write("-" * 50 + "\n")
        tf.write(f"           AgriSat Comprehensive Telemtry Report          \n")
        tf.write("-" * 50 + "\n")
        tf.write(f"Target Farm ID: {farm.farmID}\n")
        tf.write(f"Crop: {farm.cropType}\n")
        tf.write(f"Geo Boundary: {farm.geoBoundary}\n")
        tf.write(f"Report Generated at: {date.today().isoformat()}\n")
        tf.write("-" * 50 + "\n")
        tf.write(f"Assessment: {trendReport['assessment']}\n")
        tf.write(f"Net Slope: {trendReport['overall_slope']:.4f}\n")
        tf.write(f"Trend Vector: {trendReport['trend_vector']}\n")
        tf.write("-" * 60 + "\n")
        tf.write(f"Spatial Grid Diagnostic: ({quadCode})\n")
        tf.write(f"Sector Status: {zonalReport['status']}\n")
        tf.write(f"Mean NDVI: {zonalReport['mean']:.4f}\n")
        tf.write(f"Peak NDVI: {zonalReport['max']:.4f}\n")
        tf.write(f"Lowest NDVI: {zonalReport['min']:.4f}\n")
        tf.write(f"Valid Pixels: {zonalReport['coverage_pct']:.1f}%\n")
        tf.write("-" * 50 +"\n")
        tf.write("End of Generated Satellite Telemetry Document Summary. \n")
    return basePath
