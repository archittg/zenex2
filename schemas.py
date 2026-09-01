# ============================================================
# ORBITAL EDGE INTELLIGENCE
# schemas.py
# ============================================================

from typing import List, Optional, Dict

from pydantic import BaseModel, Field


# ============================================================
# BASIC YOLO DETECTION
# ============================================================

class Detection(BaseModel):
    """
    A single YOLO detection.

    confidence is the FINAL confidence displayed/used by the
    application. It is intentionally not labeled as an altered
    or modified confidence anywhere in the UI.
    """

    type: str

    # Original YOLO class/type
    raw_type: str = ""

    confidence: float = Field(
        ge=0.0,
        le=1.0
    )

    bbox: List[int]

    source_image: str = ""

    crop_url: str = ""

    annotated_image_url: str = ""


# ============================================================
# OLD / LEGACY AI RESULT
# ============================================================
#
# Kept because main.py and simulator.py still use this model.
#

class AIResult(BaseModel):

    image_id: str

    timestamp: str = ""

    detections: List[Detection] = Field(
        default_factory=list
    )

    cloud_cover: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0
    )

    raw_size_bytes: int = 0


# ============================================================
# MISSION DECISION
# ============================================================
#
# Used by the older /process-image endpoint.
#

class MissionDecision(BaseModel):

    detection_type: str

    confidence: float = Field(
        ge=0.0,
        le=1.0
    )

    priority: float

    decision: str


# ============================================================
# SATELLITE SIMULATION STATE
# ============================================================

class SimulationState(BaseModel):

    battery_percent: float

    bandwidth_available_kbps: float

    cpu_load_percent: float

    ground_link_active: bool

    raw_data_size_bytes: int

    metadata_size_bytes: int

    energy_used: float

    transmission_time_ms: float


# ============================================================
# DECISION / SCHEDULER ITEM
# ============================================================

class DecisionItem(BaseModel):

    rank: int = 0

    source_image: str = ""

    raw_type: str = ""

    type: str = ""

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0
    )

    priority: str = ""

    score: float = 0.0

    data_package: str = ""

    data_cost_kb: float = 0.0

    decision: str = ""

    bbox: List[int] = Field(
        default_factory=list
    )

    crop_url: str = ""

    reason: str = ""


# ============================================================
# IMAGE-LEVEL REPORT
# ============================================================

class ImageReport(BaseModel):

    image_id: str

    raw_size_bytes: int

    width: int

    height: int

    annotated_image_url: str

    detections: List[Detection] = Field(
        default_factory=list
    )

    sent: List[DecisionItem] = Field(
        default_factory=list
    )

    discarded: List[DecisionItem] = Field(
        default_factory=list
    )


# ============================================================
# MISSION SUMMARY
# ============================================================

class MissionSummary(BaseModel):

    mission_mode: str

    bandwidth_budget_kb: int

    images_processed: int

    total_detections: int

    transmitted: int

    discarded: int

    raw_bytes: int

    optimized_bytes: int

    raw_kb: float

    optimized_kb: float

    bandwidth_reduction_percent: float

    raw_transmission_ms: float

    optimized_transmission_ms: float

    latency_reduction_percent: float

    raw_energy_j: float

    optimized_energy_j: float

    energy_reduction_percent: float

    inference_ms: float

    packet_kb_by_type: Dict[str, float] = Field(
        default_factory=dict
    )

    battery_percent: float

    cpu_load_percent: float

    ground_link_active: bool


# ============================================================
# FINAL API RESPONSE
# ============================================================

class MissionResponse(BaseModel):

    summary: MissionSummary

    images: List[ImageReport] = Field(
        default_factory=list
    )

    sent: List[DecisionItem] = Field(
        default_factory=list
    )

    discarded: List[DecisionItem] = Field(
        default_factory=list
    )