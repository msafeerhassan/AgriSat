from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import numpy as np
from datetime import date
import os, json, pickle
import matplotlib.pyplot as plt
from sentinelhub import SHConfig, SentinelHubRequest, DataCollection, BBox, CRS, MimeType
from shapely.geometry import box, polygon

@dataclass
class FarmWorkspace:
    farmID: str
    cropType: str
    geoBoundary: Tuple[float, float, float, float]
    historicalDates: List[date] = field(default_factory=list)

    redBands: Dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    nIRbands: Dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    cloudMask: Dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    def getValidatedBounds(self) -> dict:
        return validateAndStandardizeBBox(self.geoBoundary)

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

def exportNDVIHeatMap(farmID:str, targetDateStr: str, ndviMatrix:np.ndarray, outputDir: str = "outputPlots") -> Optional[str]:
    try:
        if not os.path.exists(outputDir):
            os.makedirs(outputDir)
        filePath = os.path.join(outputDir, f"{farmID}_{targetDateStr}_Heatmap.png")
        plt.figure(figsize=(7, 6))

        cax = plt.imshow(ndviMatrix, cmap="RdYlGn", vmin=0.1, vmax=1.0, origin="upper")
        cbar = plt.colorbar(cax, orientation="vertical", pad=0.05)
        cbar.set_label('NDVI Intensity', rotation=270, labelpad=15)
        plt.title(f"AgriSat Spatial Imagery Heatmap\nProfile: {farmID} | Frame: {targetDateStr}", fontsize=12, pad=10)
        plt.xlabel("Grid Column Axis (Slices 0-9)", fontsize = 10)
        plt.ylabel("Grid Row Axis (Slices 0-9)", fontsize=10)

        plt.xticks(np.arange(10))
        plt.yticks(np.arange(10))
        plt.grid(True, which="both", color="black", linestyle=":", linewidth=0.5, alpha=0.5)

        plt.savefig(filePath, dpi=150, bbox_inches="tight")
        plt.close()
        return filePath
    except Exception as e:
        print(f"Matplotlib Error: {e}")
        return None

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

def analyzeZSG(ndviMatrix: np.ndarray, targetedQuadrant: str = "ALL") -> dict:
    quadrantDefinitons = {
        "NW": (slice(0,5), slice(0, 5)),
        "NE": (slice(0,5), slice(5, 10)),
        "SW": (slice(5,10), slice(0, 5)),
        "SE": (slice(5,10), slice(5, 10)),
    }

    combinedZonalRep = {}

    for quadCode, (rowSlice, colSlice) in quadrantDefinitons.items():
        subMatrix = ndviMatrix[rowSlice, colSlice]

        validValues = subMatrix[~np.isnan(subMatrix)]

        if validValues.size == 0:
            meanNDVI = 0.0
            status = "Insufficient Data"
        else:
            meanNDVI = float(np.mean(validValues))
            if meanNDVI >= 0.6:
                status = "Optimal Dense Canopy"
            elif meanNDVI >= 0.35:
                status = "Moderate Vegetation"
            else:
                status = "Critical Condition!"
        
        combinedZonalRep[quadCode] = {
            "mean_ndvi": meanNDVI,
            "status": status,
            "active_pixels": int(validValues.size),
            "coverage_pct": float((validValues.size / 25.0) * 100.0)
        }

    if targetedQuadrant in quadrantDefinitons:
        return combinedZonalRep[targetedQuadrant]
    return combinedZonalRep

def genHistoricalRep(farm: FarmWorkspace) -> Dict:

    if len(farm.historicalDates) < 2:
        return {
            "overall_slope": 0.0,
            "trend_vector": "Stable / Insufficient Data",
            "assessment": "Stable"
        }
    
    ndviMeans = []

    for d in farm.historicalDates:
        dateStr = d.isoformat()
        ndvi = calculateNDVI(farm.redBands[dateStr], farm.nIRbands[dateStr], farm.cloudMask[dateStr])
        valid = ndvi[~np.isnan(ndvi)]
        ndviMeans.append(float(np.mean(valid)) if valid.size > 0 else 0.0)

    cleanNDVIMeans = smoothenTemporalNDVI(ndviMeans, windowSize=3)

    xVals = np.arange(len(cleanNDVIMeans))

    slope, _ = np.polyfit(xVals, cleanNDVIMeans, 1)

    vectorSegments = []

    for val in cleanNDVIMeans:
        vectorSegments.append(f"{val:.2f}")

    if slope > 0.02:
        direction = "Upward Growth"
        assessment = "Optimal Health Progress"
    elif slope <-0.02:
        direction = "Downwards Growth"
        assessment = "Decline"
    else:
        direction = "Stable!"
        assessment = "Stable Mainatainance"

    trendVectorString = f"{direction} (" + " -> ".join(vectorSegments) + ")"

    return {
        "overall_slope": float(slope),
        "trend_vector": trendVectorString,
        'assessment': assessment
    }

def exportDetailedFarmReport(farm: FarmWorkspace, trendReport: dict, zonalReport: dict, outputDir: str = "outputReports") -> str:
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    basePath = os.path.join(outputDir, f"{farm.farmID}_Summary")
    jsonPayload = {
        "farmID": farm.farmID,
        "cropType": farm.cropType,
        "geographical_bounds": {
            "min_x": farm.geoBoundary[0],
            "min_y": farm.geoBoundary[1],
            "max_x": farm.geoBoundary[2],
            "max_y": farm.geoBoundary[3]
        },
        "temporalTrends": {
            "assessment": trendReport.get("assessment", "Unknown"),
            "overall_slope": float(trendReport.get("overall_slope", 0.0)),
            "trend_vector": trendReport.get("trend_vector", "Stable")
        },
        "spatial_zones": zonalReport,
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
        tf.write(f"Vector Trajectory: {trendReport.get('trend_vector', 'N/A')}\n")
        tf.write(f"Net Slope: {trendReport.get('overall_slope', 0.0):.6f}\n")
        tf.write(f"Health Assessment: {trendReport.get('assessment', 'N/A')}\n")
        tf.write("-" * 50 + "\n")
        tf.write(f"Spatial Grid Diagnostic\n")

        for quad, metrics in zonalReport.items():
            tf.write(f"Sector: {quad}\n")
            tf.write(f"Mean NDVI: {metrics['mean_ndvi']:.4f}\n")
            tf.write(f"Clean Pixel Count: {metrics['active_pixels']}/25\n")
            tf.write(f"Canopy Density: {metrics['coverage_pct']}%\n")
            tf.write(f"Sector Status: {metrics['status']}\n")
        tf.write("-" * 50 +"\n")
    return basePath

def verifySentinelCredentials() -> bool:
    clientID = os.getenv("CLIENT_ID")
    clientSecret = os.getenv("CLIENT_SECRET")

    if not clientID or not clientSecret:
        print("\n Credentials Missing :(")
        return False

    try:
        config = SHConfig()
        config.sh_client_id = clientID
        config.sh_client_secret = clientSecret

        if config.sh_client_id and config.sh_client_secret:
            print("Sentinel Hub SDK Config Initialized")
            return True
    except Exception as e:
        print(f"Error in Sentinel Hub Credentials Initialization: {e}")
        return False

def validateAndStandardizeBBox(bboxTuple: Tuple[float, float, float, float] , crsFormat: str = "EPSG:4326") -> dict:
    if len(bboxTuple) != 4:
        raise ValueError("Geospatial Boundaries are supposed to be 4-tuple sequence of floats")

    minX, minY, maxX, maxY = bboxTuple

    if minX >= maxX or minY >= maxY:
        raise ValueError(f"Invalid Bounding Box Dimensions : {bboxTuple}")
    
    spatialEnv = box(minX, minY, maxX, maxY)

    if not spatialEnv.is_valid or spatialEnv.is_empty:
        raise ValueError("Coordinates can't form valid spatial polygon")
    
    return {
        "bbox": (minX, minY, maxX, maxY),
        "crs" : crsFormat,
        "area_deg_sq": spatialEnv.area
    }

def genSentinelNDVIReq(farm: FarmWorkspace, targetDate: date) -> SentinelHubRequest:
    clientID = os.getenv("CLIENT_ID")
    cleintSecret = os.getenv("CLIENT_SECRET")

    config = SHConfig()
    config.sh_client_id = clientID
    config.sh_client_secret = cleintSecret

    spatialMeta = farm.getValidatedBounds()
    minX, minY, maxX, maxY = spatialMeta['bbox']

    sentialBbox = BBox(bbox=[minX, minY, maxX, maxY], crs=CRS.WGS84)

    evalScriptPayLoad = """
    function setup() {
        return {
            input: [
                "B04",
                "B08",
                "SCL"
            ],
            output: {
                bands: 3,
                sampleType: "FLOAT32"
            }
        };
    }

    function evaluatePixel(sample){
        return [sample.B04, sample.B08, sample.SCL]
    }
    """
    request = SentinelHubRequest(
        evalscript=evalScriptPayLoad,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=(f"{targetDate.isoformat()}T00:00:00Z", f"{targetDate.isoformat()}T23:59:59Z")
            )
        ],
        responses=[
            SentinelHubRequest.output_response('default', MimeType.TIFF)
        ],
        bbox=sentialBbox,
        config=config
    )

    return request

def verifyAPIConnectionMock(farm: FarmWorkspace) -> bool:
    try:
        testDate = date(2026, 6, 15)
        req = genSentinelNDVIReq(farm, testDate)

        if isinstance(req, SentinelHubRequest):
            print("Mock Test Passed")
            return True
    except Exception as mockErr:
        print(f"Error: {mockErr}")
        return False
    return False

def processSatelliteResponseMatrix(rawApiResponseData : np.ndarray, targetShape: Tuple[int, int] = (10, 10)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if rawApiResponseData.ndim < 3 or rawApiResponseData.shape[-1] != 3:
        if rawApiResponseData.shape[0] == 3:
            rawApiResponseData = np.moveaxis(rawApiResponseData, 0, -1)

    rawRed = rawApiResponseData[..., 0]
    rawNir = rawApiResponseData[..., 1]
    rawScl = rawApiResponseData[..., 2]

    srcH, srcW = rawRed.shape
    tgtH, tgtW = targetShape

    rowIndices = np.linspace(0, srcH - 1, tgtH).astype(int)
    colIndices = np.linspace(0, srcW - 1, tgtW).astype(int)

    resizedRed = rawRed[np.ix_(rowIndices, colIndices)]
    resizedNir = rawNir[np.ix_(rowIndices, colIndices)]
    resizedScl = rawScl[np.ix_(rowIndices, colIndices)]

    if np.max(resizedRed) > 1.0:
        normalizedRed = np.clip(resizedRed / 10000, 0.0, 1.0)
    else:
        normalizedRed = np.clip(resizedRed, 0.0, 1.0)

    if np.max(resizedNir) > 1.0:
        normalizedNir = np.clip(resizedNir / 10000, 0.0, 1.0)
    else:
        normalizedNir = np.clip(resizedNir, 0.0, 1.0)
    
    cloudMask = (resizedScl == 3) | (resizedScl == 8) | (resizedScl == 9) | (resizedScl == 10)

    return normalizedRed, normalizedNir, cloudMask

def verifyMatrixReshaping() -> bool:
    print("Response Parsing and Reshaping Test")

    mockHighRes = np.zeros((150, 150, 3))
    mockHighRes[..., 0] = np.random.uniform(400, 1200, size=(150,150))
    mockHighRes[..., 0] = np.random.uniform(5000, 7500, size=(150, 150))
    mockHighRes[..., 2] = 9.0

    try:
        redGrid, nirGrid, cloudGrid = processSatelliteResponseMatrix(mockHighRes, targetShape=(10, 10))
        
        shapeCorrect = (redGrid.shape == (10, 10) and nirGrid.shape == (10, 10) and cloudGrid.shape == (10, 10))
        boundsCorrect = (np.max(redGrid) <= 1.0 and np.max(nirGrid) <= 1.0)
        maskCorrect = (cloudGrid[0, 0] == True)

        if shapeCorrect and boundsCorrect and maskCorrect:
            print("Successfully Verified Matrix Reshaping")
            return True
        else:
            print("Matrix Reshaping Verification Failed")
            return False
    
    except Exception as e:
        print(f"Execution Failed: {e}")
        return False

def downloadAndRegisterSatelliteTelemetry(farm: FarmWorkspace, targetDate: date) -> bool:
    print(f"LIVE API REQUEST: Contacting Sentinel Hub for {farm.farmID} ({targetDate.isoformat()})")

    try:
        request = genSentinelNDVIReq(farm, targetDate)

        rawDownloadDataList = request.get_data()

        if not rawDownloadDataList or len(rawDownloadDataList) == 0:
            print("No Imagery Found for this")
            return False
        
        rawImageryMatrix = np.array(rawDownloadDataList[0])

        normRed, normNir, cloudMask = processSatelliteResponseMatrix(rawImageryMatrix, targetShape=(10, 10))

        farm.addTelemetrySnapshot(targetDate, normRed, normNir, cloudMask)

        serializeFarmWorkspace(farm)

        print(f"Telemetry Data Successfuly Commited to database of {farm.farmID}")
        return True
    except Exception as apiExecErr:
        print(f"Error: {apiExecErr}")
        return False
    
def smoothenTemporalNDVI(rawMeans: List[float], windowSize: int = 3) -> List[float]:
    if len(rawMeans) < windowSize:
        return rawMeans

    smoothedList = []
    extendedBounds = windowSize // 2

    paddedData = np.pad(rawMeans, extendedBounds, mode='edge')

    for i in range(len(rawMeans)):
        windowSlice = paddedData[i : i + windowSize]
        smoothedList.append(float(np.mean(windowSlice)))
    
    return smoothedList

def predictFutureNDVI(cleanMeans: List[float], projectionSteps: int = 3) -> List[float]:
    if len(cleanMeans) < 2:
        if cleanMeans:
            return cleanMeans[-1] * projectionSteps
        else:
            return 0.0
        
    xVals = np.arange(len(cleanMeans))

    slope, intercept = np.polyfit(xVals, cleanMeans, 1)

    predictions = []

    for step in range(1, projectionSteps + 1):
        futureIdx = len(cleanMeans) - 1 + step
        projectVal = slope * futureIdx + intercept

        clampedVal = max(-0.1, min(1.0, float(projectVal)))
        predictions.append(clampedVal)
    return predictions