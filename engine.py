from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import numpy as np
from datetime import date

@dataclass
class FarmWorkspace:
    farmID: str
    cropType: str
    geoBoundary: Tuple[float, float, float, float]
    historialDates: List[date] = field(default_factory=list)

    redBands: Dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    nIRbands: Dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    cloudMask: Dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    def addTelemetrySnapshot(self, snapShotDate: date, redBands: np.ndarray, nIRBands: np.ndarray, cloudMask: np.ndarray):
        dateStr = snapShotDate.isoformat()
        if snapShotDate not in self.historialDates:
            self.historialDates.append(snapShotDate)
            self.historialDates.sort()

        self.redBands[dateStr] = redBands
        self.nIRbands[dateStr] = nIRBands
        self.cloudMask[dateStr] = cloudMask

def genSpectralBand(condition: str, bandType: str, shape: Tuple[int, int] = (10, 10)) -> np.ndarray:
    if condition.lower() == "healthy":
        redLow, redHigh = 0.05, 0.15
        nIRLow, nIRHigh = 0.60, 0.85
    elif condition.lower() == "stressed":
        redLow, redHigh = 0.60, 0.85
        nIRLow, nIRHigh = 0.05, 0.15
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