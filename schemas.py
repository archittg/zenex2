# backend/schemas.py
from pydantic import BaseModel
from typing import List, Optional

class Detection(BaseModel):
    type: str
    confidence: float
    bbox: List[int]

class AIResult(BaseModel):
    image_id: str
    timestamp: str
    detections: List[Detection]
    cloud_cover: float
    raw_size_bytes: int

class MissionDecision(BaseModel):
    detection_type: str
    confidence: float
    priority: float
    decision: str  # "TRANSMIT", "COMPRESS", "STORE", "DISCARD"

class SimulationState(BaseModel):
    battery_percent: float
    bandwidth_available_kbps: float
    cpu_load_percent: float
    ground_link_active: bool
    raw_data_size_bytes: int
    metadata_size_bytes: int
    energy_used: float
    transmission_time_ms: float