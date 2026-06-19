from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import numpy as np
from datetime import date
import os, json, pickle, time
from datetime import timedelta
import matplotlib.pyplot as plt
from sentinelhub import SHConfig, SentinelHubRequest, DataCollection, BBox, CRS, MimeType, SentinelHubDownloadClient
from shapely.geometry import box, Polygon, Point

GRID_RESOLUTION: Tuple[int, int] = (15, 15)

def fetchWithRetry(requestFn, maxAttempts: int = 3, baseDelaySeconds: float = 2.0):
    lastError = None
    for attempt in range(1, maxAttempts + 1):
        try:
            return requestFn()
        except Exception as fetchErr:
            lastError = fetchErr
            if attempt < maxAttempts:
                sleepFor = baseDelaySeconds * (2 ** (attempt - 1))
                print(f"Attempt {attempt}/{maxAttempts} failed: {fetchErr}. Retrying in {sleepFor:.1f}s...")
                time.sleep(sleepFor)
            else:
                print(f"ALL {maxAttempts} ATTEMPTS FAILED :(")
    
    raise lastError

@dataclass
class FarmWorkspace:
    farmID: str
    cropType: str
    geoBoundary: Tuple[float, float, float, float]
    polygonCoords: List[Tuple[float, float]]
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

def genPolygonRasterMask(farm: FarmWorkspace, targetShape: tuple[int, int]) -> np.ndarray:
    height, width = targetShape
    mask = np.zeros((height, width), dtype=bool)

    if not farm.polygonCoords:
        return np.ones((height, width), dtype=bool)
    
    polyGeom = Polygon(farm.polygonCoords)
    minX, minY, maxX, maxY = farm.geoBoundary

    buffer = 0.00001

    xCoords = np.linspace(minX - buffer, maxX + buffer, width)
    yCoords = np.linspace(maxY + buffer, minY - buffer, height)

    for r in range(height):
        for c in range(width):
            pt = Point(xCoords[c], yCoords[r])
            if polyGeom.contains(pt) or polyGeom.intersects(pt):
                mask[r, c] = True
            
    if np.sum(mask) < 5:
        print("Weak Polygon Mask Detected")
        mask = np.ones((height, width), dtype=bool)
    
    print(f"Polygon Mask Created: {np.sum(mask)}/{height*width} pixels inside boundary")
    return mask

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


def genCloudMaskFromSCL(sclArray: np.ndarray) -> np.ndarray:
    noDataPixels = (sclArray == 0)
    cloudPixels = (sclArray == 3) | (sclArray == 8) | (sclArray == 9) | (sclArray == 10)
    return cloudPixels | noDataPixels

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
        "polygonCoords": farm.polygonCoords,
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
        geoBoundary=tuple(metaData["geoBoundary"]),
        polygonCoords=metaData.get("polygonCoords", [])
    )

    farm.historicalDates = [date.fromisoformat(d) for d in metaData["historicalDates"]]

    with open(arrayPath, "rb") as f:
        arrays = pickle.load(f)
        farm.redBands = arrays["redBands"]
        farm.nIRbands = arrays["nirBands"]
        farm.cloudMask = arrays["cloudMasks"]
    return farm

def analyzeZSG(ndviMatrix: np.ndarray, targetedQuadrant: str = "ALL") -> dict:
    
    height, width = ndviMatrix.shape
    midRow = height // 2
    midCol = width // 2

    quadrantDefinitons = {
        "NW": (slice(0,midRow), slice(0, midCol)),
        "NE": (slice(0,midRow), slice(midCol, width)),
        "SW": (slice(midRow,height), slice(0, midCol)),
        "SE": (slice(midRow,height), slice(midCol, width)),
    }

    combinedZonalRep = {}

    for quadCode, (rowSlice, colSlice) in quadrantDefinitons.items():
        subMatrix = ndviMatrix[rowSlice, colSlice]

        validValues = subMatrix[~np.isnan(subMatrix)]
        totalQuadrantPixels = subMatrix.size

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
            "total_pixels": int(totalQuadrantPixels),
            "coverage_pct": float((validValues.size / totalQuadrantPixels) * 100.0)
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
            tf.write(f"Clean Pixel Count: {metrics['active_pixels']}/{metrics.get("total_pixels", metrics['active_pixels'])}\n")
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

def verifyLiveSentinelCredentials(clientID: str, clientSecret: str) -> Tuple[bool, str]:
    if not clientID or not clientSecret:
        return False, "Credentials Missing"
    
    try:
        config = SHConfig()
        config.sh_client_id = clientID
        config.sh_client_secret = clientSecret

        client = SentinelHubDownloadClient(config=config)
        client.get_json(
            url="https://services.sentinel-hub.com/oauth/token",
            post_values={
                "grant_type": "client_credentials",
                "client_id": clientID,
                "client_secret": clientSecret,
            },
        )
        return True, "Connected"
    except Exception as authErr:
        return False, f"Authentication Failed: {authErr}"

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

def genSentinelNDVIReq(farm: FarmWorkspace, targetDate: date, config: SHConfig, windowDays: int = 4) -> SentinelHubRequest:
    minLon, minLat, maxLon, maxLat = farm.geoBoundary
    
    sentinelBBox = BBox(bbox=[minLon, minLat, maxLon, maxLat], crs=CRS.WGS84)

    windowStart = (targetDate - timedelta(days=windowDays)).isoformat()
    windowEnd = (targetDate + timedelta(days=windowDays)).isoformat()

    dateWindow = (windowStart, windowEnd)

    evalScript = """
    function setup() {
        return {
            input: [{
                bands: [
                    "B04",
                    "B08",
                    "SCL"
                ],
                units: [
                    "REFLECTANCE",
                    "REFLECTANCE",
                    "DN"
                ]
            }],
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
        evalscript=evalScript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=dateWindow,
                mosaicking_order = 'leastCC'
            )
        ],
        responses=[
            SentinelHubRequest.output_response('default', MimeType.TIFF)
        ],
        bbox=sentinelBBox,
        size=GRID_RESOLUTION,
        config=config
    )

    return request

def verifyMatrixReshaping(farm: FarmWorkspace) -> bool:
    try:
        minLon, minLat, maxLon, maxLat = farm.geoBoundary
        testBbox = BBox(bbox=[minLon, minLat, maxLon, maxLat], crs=CRS.WGS84)
        return testBbox.is_valid()
    except Exception:
        return False

def downloadAndRegisterSatelliteTelemetry(farm: FarmWorkspace, clientID: str, clientSecret:str) -> dict:
    perDateResults = []
    if not clientID or not clientSecret:
        print("SentinelHub Credentials Missing :(")
        return {
            "success": False,
            "successCount": 0,
            "totalAttempted": 0,
            "perDate": perDateResults
        }

    config = SHConfig()
    config.instance_id = clientID
    config.sh_client_id = clientID
    config.sh_client_secret = clientSecret

    polyMask = genPolygonRasterMask(farm, GRID_RESOLUTION)
    successCount = 0
    print(f"Starting data Download from Sentinel 2 for {farm.farmID}")

    lookBackDays = list(range(30, 0, -5))

    for lookBack in lookBackDays:
        snapDate = date.today() - timedelta(days=lookBack)
        dateStr = snapDate.isoformat()
        try:
            request = genSentinelNDVIReq(farm, snapDate, config)
            rawDataList = fetchWithRetry(lambda: request.get_data())

            if not rawDataList or len(rawDataList) == 0:
                print(f"No Telemetry Data Found for Date: {snapDate}")
                perDateResults.append({
                    "date": dateStr,
                    "status": "no_data"
                })
                continue
            
            matrixFrame = rawDataList[0]

            redArr = matrixFrame[:, :, 0].astype(float)
            nirArr = matrixFrame[:, :, 1].astype(float)
            sclArr = matrixFrame[:, :, 2]

            redArr = np.clip(redArr, 0.0, 1.0)
            nirArr = np.clip(nirArr, 0.0, 1.0)

            cloudPixels = genCloudMaskFromSCL(sclArr)
            noDataPixels = (sclArr == 0)

            if np.all(noDataPixels):
                print(f"Entire Scene is NO Data for {snapDate}")
                perDateResults.append({
                    "date": dateStr,
                    "status": "no_data"
                })
                continue
            totalPixels = GRID_RESOLUTION[0] * GRID_RESOLUTION[1]
            cloudPct = float(np.sum(cloudPixels) / totalPixels)
            if cloudPct > 0.70:
                print(f"Too Cloudy on {snapDate} ({cloudPct:.1%})")
                perDateResults.append({
                    "date": dateStr,
                    "status": "too_cloudy",
                    "cloudPct": cloudPct
                })
                continue
            invalidPixels = cloudPixels | (~polyMask)
            redArr[invalidPixels] = np.nan
            nirArr[invalidPixels] = np.nan

            farm.addTelemetrySnapshot(snapDate, redArr, nirArr, invalidPixels)
            successCount += 1
            ndviTest = calculateNDVI(redArr, nirArr, invalidPixels)
            meanNDVI = float(np.nanmean(ndviTest))

            print(f"Successfully Fetched Sentinel Data for {farm.farmID} of date {snapDate} - Mean NDVI: {meanNDVI:.4f} | Valid Pixels: {np.sum(~np.isnan(ndviTest))}")
            perDateResults.append({
                "date": dateStr,
                "status": "fetched",
                "meanNDVI": meanNDVI,
                "cloudPct": cloudPct
            })
        except Exception as e:
            print(f"SentinelHub Download Failed on {snapDate}: {e}")
            perDateResults.append({
                "date": dateStr,
                "status": "error",
                "message": str(e)
            })
            continue
    print(f"Download Complete: {successCount} successfull Snapshots Fetched")
    return{
        "success": successCount > 0,
        "successCount": successCount,
        "totalAttempted": len(lookBackDays),
        "perDate": perDateResults
    }
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
            return [float(cleanMeans[-1])] * projectionSteps
        else:
            return [0.0] * projectionSteps
        
    xVals = np.arange(len(cleanMeans))

    slope, intercept = np.polyfit(xVals, cleanMeans, 1)

    predictions = []

    for step in range(1, projectionSteps + 1):
        futureIdx = len(cleanMeans) - 1 + step
        projectVal = slope * futureIdx + intercept

        # clampedVal = max(-0.1, min(1.0, float(projectVal)))
        predictions.append(float(np.clip(projectVal, -1.0, 1.0)))
    return predictions

def parseGeoJSONPolygon(filePath: str) -> Optional[Tuple[str, str, Tuple[float, float, float, float], List[Tuple[float, float]]]]:
    if not os.path.exists(filePath):
        print(f"Error: Target file doesn't exists: {filePath}")
        return None
    try:
        with open(filePath, 'r') as file:
            geoData = json.load(file)

        feature = geoData["features"][0]
        properties = feature["properties"]
        geometry = feature["geometry"]

        farmID = properties.get("farmID", "FARM-UNKNOWN")
        cropType = properties.get("cropType", "Unknown Crop")

        if geometry["type"] != "Polygon":
            print("Error: Vector File must have valid Polygon Geometry Feature")
            return None
        
        rawCoordinates = geometry["coordinates"][0]

        polygonCoords = [(float(pt[0]), float(pt[1])) for pt in rawCoordinates]

        tempPoly = Polygon(polygonCoords)

        geoBoundary = tempPoly.bounds

        return farmID, cropType, geoBoundary, polygonCoords
    except Exception as e:
        print(f"Failed parsing GeoJSON File: {e}")
        return None