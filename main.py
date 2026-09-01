# ============================================================
# ORBITAL EDGE INTELLIGENCE
# main.py
# ============================================================

import json
import time
import traceback
from pathlib import Path
from typing import List

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    Request,
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
)
from fastapi.staticfiles import StaticFiles

from detector import analyze_image
from simulator import satellite


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Orbital Edge Intelligence",
    version="4.0.0",
    description=(
        "AI-powered onboard satellite detection, "
        "mission-aware evidence selection and "
        "autonomous downlink optimisation."
    ),
)

# ============================================================
# GLOBAL ERROR HANDLER
# ============================================================

@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request,
    exc: Exception,
):
    print()
    print("=" * 80)
    print("!!! UNHANDLED BACKEND ERROR !!!")
    print("=" * 80)
    print(
        f"REQUEST: {request.method} {request.url}"
    )
    print(
        f"ERROR: {type(exc).__name__}: {exc}"
    )
    traceback.print_exc()
    print("=" * 80)
    print()

    return JSONResponse(
        status_code=500,
        content={
            "detail":
                f"{type(exc).__name__}: {str(exc)}",
        },
    )


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# DIRECTORIES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

RUNTIME = BASE_DIR / "runtime"
UPLOADS = RUNTIME / "uploads"

RUNTIME.mkdir(
    parents=True,
    exist_ok=True,
)

UPLOADS.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# STATIC RUNTIME FILES
# ============================================================

app.mount(
    "/runtime",
    StaticFiles(
        directory=str(RUNTIME)
    ),
    name="runtime",
)


# ============================================================
# MISSION MODES
# ============================================================

MISSION_MODES = {
    "MARITIME_SURVEILLANCE":
        "Maritime Surveillance",

    "URBAN_MONITORING":
        "Urban Monitoring",

    "DISASTER_RESPONSE":
        "Disaster Response",

    "WILDFIRE_RESPONSE":
        "Wildfire Response",

    "GENERAL":
        "General Observation",
}


# ============================================================
# YOLO CLASS NORMALIZATION
# ============================================================

CLASS_MAP = {

    # -------------------------
    # MARITIME
    # -------------------------

    "boat":
        "ship",

    "ship":
        "ship",

    # -------------------------
    # URBAN VEHICLES
    # -------------------------

    "car":
        "vehicle",

    "truck":
        "vehicle",

    "bus":
        "vehicle",

    "motorcycle":
        "vehicle",

    # -------------------------
    # URBAN OBJECTS
    # -------------------------

    "traffic light":
        "traffic_light",

    "stop sign":
        "stop_sign",

    "fire hydrant":
        "fire_hydrant",

    "person":
        "person",

    "bicycle":
        "bicycle",

    # -------------------------
    # OTHER
    # -------------------------

    "airplane":
        "aircraft",

    "train":
        "train",
}


# ============================================================
# MISSION RELEVANCE
# ============================================================

MISSION_RULES = {

    # ========================================================
    # MARITIME
    # ========================================================

    "MARITIME_SURVEILLANCE": {

        "ship":
            {
                "priority": "HIGH",
                "weight": 1.00,
            },

    },


    # ========================================================
    # URBAN
    # ========================================================

    "URBAN_MONITORING": {

        "vehicle":
            {
                "priority": "HIGH",
                "weight": 1.00,
            },

        "person":
            {
                "priority": "MEDIUM",
                "weight": 0.65,
            },

        "traffic_light":
            {
                "priority": "MEDIUM",
                "weight": 0.60,
            },

        "bicycle":
            {
                "priority": "LOW",
                "weight": 0.30,
            },

        "stop_sign":
            {
                "priority": "LOW",
                "weight": 0.30,
            },

        "fire_hydrant":
            {
                "priority": "LOW",
                "weight": 0.25,
            },

    },


    # ========================================================
    # DISASTER RESPONSE
    # ========================================================

    "DISASTER_RESPONSE": {

        "person":
            {
                "priority": "CRITICAL",
                "weight": 1.00,
            },

        "vehicle":
            {
                "priority": "HIGH",
                "weight": 0.85,
            },

        "ship":
            {
                "priority": "HIGH",
                "weight": 0.85,
            },

        "aircraft":
            {
                "priority": "HIGH",
                "weight": 0.85,
            },

    },


    # ========================================================
    # WILDFIRE
    # ========================================================

    "WILDFIRE_RESPONSE": {

        "person":
            {
                "priority": "HIGH",
                "weight": 0.85,
            },

        "vehicle":
            {
                "priority": "MEDIUM",
                "weight": 0.60,
            },

        "aircraft":
            {
                "priority": "MEDIUM",
                "weight": 0.60,
            },

    },


    # ========================================================
    # GENERAL
    # ========================================================

    "GENERAL": {

        "ship":
            {
                "priority": "MEDIUM",
                "weight": 0.60,
            },

        "vehicle":
            {
                "priority": "MEDIUM",
                "weight": 0.60,
            },

        "person":
            {
                "priority": "MEDIUM",
                "weight": 0.60,
            },

        "traffic_light":
            {
                "priority": "LOW",
                "weight": 0.30,
            },

        "aircraft":
            {
                "priority": "MEDIUM",
                "weight": 0.60,
            },

        "train":
            {
                "priority": "MEDIUM",
                "weight": 0.60,
            },
    },
}


# ============================================================
# HISTORY
# ============================================================

HISTORY = []


# ============================================================
# HELPERS
# ============================================================

def normalize_class(
    raw_class: str
) -> str:

    value = (
        str(raw_class)
        .strip()
        .lower()
    )

    return CLASS_MAP.get(
        value,
        value,
    )


def mission_rule(
    object_type: str,
    mission_mode: str,
):

    rules = MISSION_RULES.get(
        mission_mode,
        {},
    )

    return rules.get(
        object_type
    )


def calculate_score(
    confidence: float,
    priority_weight: float,
) -> float:

    # Mission importance has greater weight than
    # perception confidence.

    score = (
        0.40 * confidence
        +
        0.60 * priority_weight
    )

    return max(
        0.0,
        min(
            1.0,
            score,
        ),
    )


def choose_data_package(
    confidence: float
):

    confidence_percent = (
        confidence * 100.0
    )

    if confidence_percent >= 90:

        return (
            "METADATA_ONLY",
            1,
        )

    if confidence_percent >= 60:

        return (
            "METADATA_PLUS_THUMBNAIL",
            4,
        )

    if confidence_percent >= 30:

        return (
            "METADATA_PLUS_IMAGE_CROP",
            8,
        )

    return (
        "METADATA_PLUS_FULL_IMAGE",
        20,
    )


def make_decisions(
    detections: list,
    bandwidth_kb: int,
    mission_mode: str,
):

    candidates = []

    # ========================================================
    # STEP 1 — MISSION FILTER
    # ========================================================

    for detection in detections:

        raw_class = detection.get(
            "target_class",
            "unknown",
        )

        object_type = normalize_class(
            raw_class
        )

        confidence = float(
            detection.get(
                "confidence_score",
                0.0,
            )
        )

        confidence = max(
            0.0,
            min(
                1.0,
                confidence,
            )
        )

        rule = mission_rule(
            object_type,
            mission_mode,
        )

        # ====================================================
        # HARD REJECTION
        # ====================================================
        #
        # If the object isn't part of this mission,
        # it CANNOT consume bandwidth.
        #

        if rule is None:

            candidates.append({

                "target_class":
                    raw_class,

                "type":
                    object_type,

                "confidence":
                    confidence,

                "priority":
                    "DISCARD",

                "score":
                    0.0,

                "data_package":
                    "NONE",

                "data_cost_kb":
                    0,

                "decision":
                    "DISCARD",

                "bounding_box":
                    detection.get(
                        "bounding_box",
                        [],
                    ),

                "source_image":
                    detection.get(
                        "source_image",
                        "",
                    ),

                "crop_url":
                    detection.get(
                        "crop_url",
                        "",
                    ),

                "reason":
                    (
                        f"{raw_class} is not relevant "
                        f"to the "
                        f"{MISSION_MODES[mission_mode]} "
                        f"mission."
                    ),
            })

            continue

        # ====================================================
        # MISSION RELEVANT
        # ====================================================

        priority = rule[
            "priority"
        ]

        weight = rule[
            "weight"
        ]

        score = calculate_score(
            confidence,
            weight,
        )

        package, cost_kb = (
            choose_data_package(
                confidence
            )
        )

        candidates.append({

            "target_class":
                raw_class,

            "type":
                object_type,

            "confidence":
                round(
                    confidence,
                    4,
                ),

            "priority":
                priority,

            "score":
                round(
                    score,
                    4,
                ),

            "data_package":
                package,

            "data_cost_kb":
                cost_kb,

            "decision":
                "PENDING",

            "bounding_box":
                detection.get(
                    "bounding_box",
                    [],
                ),

            "source_image":
                detection.get(
                    "source_image",
                    "",
                ),

            "crop_url":
                detection.get(
                    "crop_url",
                    "",
                ),

            "reason":
                (
                    f"{raw_class} is relevant to "
                    f"the {MISSION_MODES[mission_mode]} "
                    f"mission."
                ),
        })

    # ========================================================
    # STEP 2 — HIGHEST VALUE FIRST
    # ========================================================

    candidates.sort(
        key=lambda item:
            item["score"],
        reverse=True,
    )

    # ========================================================
    # STEP 3 — SHARED BANDWIDTH
    # ========================================================

    remaining_kb = float(
        bandwidth_kb
    )

    sent = []
    discarded = []

    for item in candidates:

        # Already rejected by mission.
        if item["decision"] == "DISCARD":

            discarded.append(
                item
            )

            continue

        cost = float(
            item["data_cost_kb"]
        )

        if cost <= remaining_kb:

            item["decision"] = (
                "TRANSMIT"
            )

            item["reason"] = (
                f"{item['priority']} priority "
                f"target selected for downlink."
            )

            remaining_kb -= cost

            sent.append(
                item
            )

        else:

            item["decision"] = (
                "DISCARD"
            )

            item["reason"] = (
                "Mission-relevant target, but "
                "insufficient remaining pass bandwidth."
            )

            discarded.append(
                item
            )

    # ========================================================
    # RANK
    # ========================================================

    sent.sort(
        key=lambda item:
            item["score"],
        reverse=True,
    )

    discarded.sort(
        key=lambda item:
            item["score"],
        reverse=True,
    )

    for rank, item in enumerate(
        sent,
        start=1,
    ):

        item["rank"] = rank

    for rank, item in enumerate(
        discarded,
        start=1,
    ):

        item["rank"] = rank

    return {
        "sent":
            sent,

        "discarded":
            discarded,

        "remaining_bandwidth_kb":
            round(
                remaining_kb,
                2,
            ),
    }


# ============================================================
# ENERGY ESTIMATE
# ============================================================

def energy_estimate(
    data_bytes: int
) -> float:

    return (
        0.1
        +
        (
            data_bytes /
            1024.0
        )
        *
        0.05
    )


# ============================================================
# TELEMETRY
# ============================================================

def telemetry_dict():

    """
    Read satellite telemetry directly from the simulator.

    This intentionally does not depend on a get_status()
    method, because different versions of simulator.py may
    expose telemetry differently.
    """

    return {

        "battery_percent":
            float(
                getattr(
                    satellite,
                    "battery_percent",
                    getattr(
                        satellite,
                        "battery",
                        78.0
                    )
                )
            ),

        "bandwidth_available_kbps":
            float(
                getattr(
                    satellite,
                    "bandwidth_available_kbps",
                    getattr(
                        satellite,
                        "bandwidth_kbps",
                        500.0
                    )
                )
            ),

        "cpu_load_percent":
            float(
                getattr(
                    satellite,
                    "cpu_load_percent",
                    getattr(
                        satellite,
                        "cpu_load",
                        10.0
                    )
                )
            ),

        "ground_link_active":
            bool(
                getattr(
                    satellite,
                    "ground_link_active",
                    getattr(
                        satellite,
                        "ground_link",
                        True
                    )
                )
            ),

        "raw_data_size_bytes":
            int(
                getattr(
                    satellite,
                    "raw_data_size_bytes",
                    getattr(
                        satellite,
                        "total_raw_data",
                        0
                    )
                )
            ),

        "metadata_size_bytes":
            int(
                getattr(
                    satellite,
                    "metadata_size_bytes",
                    getattr(
                        satellite,
                        "total_transmitted_data",
                        0
                    )
                )
            ),

        "energy_used":
            float(
                getattr(
                    satellite,
                    "energy_used",
                    0.0
                )
            ),

        "transmission_time_ms":
            float(
                getattr(
                    satellite,
                    "transmission_time_ms",
                    0.0
                )
            ),
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status":
            "online",

        "system":
            "Orbital Edge Intelligence",

        "detector":
            "YOLOv8n",

        "mission_modes":
            list(
                MISSION_MODES.keys()
            ),
    }


# ============================================================
# SATELLITE STATUS
# ============================================================

@app.get(
    "/satellite/status"
)
def get_satellite_status():

    return telemetry_dict()


# ============================================================
# MISSION MODES
# ============================================================

@app.get(
    "/api/mission-modes"
)
def get_mission_modes():

    return {
        "modes":
            MISSION_MODES
    }


# ============================================================
# HISTORY
# ============================================================

@app.get(
    "/api/history"
)
def get_history():

    return {
        "history":
            HISTORY
    }


# ============================================================
# RESET
# ============================================================

@app.post(
    "/simulation/reset"
)
def reset_simulation():

    try:

        if hasattr(
            satellite,
            "reset",
        ):

            satellite.reset()

        else:

            satellite.battery = 78.0
            satellite.bandwidth_kbps = 500.0
            satellite.cpu_load = 10.0
            satellite.ground_link = True

            satellite.total_raw_data = 0
            satellite.total_transmitted_data = 0

            satellite.energy_used = 0.0
            satellite.transmission_time_ms = 0.0

        HISTORY.clear()

        return {
            "status":
                "success",

            "message":
                "Satellite simulation reset.",
        }

    except Exception as exc:

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# ============================================================
# ORBIT PASS
# ============================================================

@app.post(
    "/simulation/start"
)
def start_simulation():

    try:

        # Simulate a changing orbital state.

        current = telemetry_dict()

        new_battery = max(
            10.0,
            float(
                current.get(
                    "battery_percent",
                    78.0,
                )
            )
            -
            15.0,
        )

        new_cpu = 85.5

        satellite.battery_percent = (
            new_battery
        )

        satellite.cpu_load_percent = (
            new_cpu
        )

        return {
            "status":
                "success",

            "message":
                "Orbit pass simulated successfully.",
        }

    except Exception as exc:

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# ============================================================
# MAIN ANALYSIS
# ============================================================

@app.post(
    "/api/analyze"
)
async def analyze(

    files: List[
        UploadFile
    ] = File(...),

    mission_mode: str = Form(
        "MARITIME_SURVEILLANCE"
    ),

    bandwidth_kb: int = Form(
        20
    ),

    confidence_threshold: float = Form(
        0.25
    ),
):

    # ========================================================
    # VALIDATION
    # ========================================================

    if mission_mode not in MISSION_MODES:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid mission mode: "
                f"{mission_mode}"
            ),
        )

    if not (
        1 <= bandwidth_kb <= 100
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Bandwidth budget must be "
                "between 1 and 100 KB."
            ),
        )

    if not (
        0.01
        <= confidence_threshold
        <= 0.99
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Confidence threshold must be "
                "between 0.01 and 0.99."
            ),
        )

    if not files:

        raise HTTPException(
            status_code=400,
            detail = "No images uploaded.",
        )

    started = time.perf_counter()

    all_detections = []
    image_records = []

    # ========================================================
    # RUN YOLO ON EVERY IMAGE
    # ========================================================

    for index, upload in enumerate(
        files
    ):

        original_filename = (
            upload.filename
            or f"image_{index}.jpg"
        )

        extension = (
            Path(
                original_filename
            )
            .suffix
            .lower()
        )

        if extension not in {
            ".jpg",
            ".jpeg",
            ".png",
        }:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Unsupported image: "
                    f"{original_filename}"
                ),
            )

        safe_filename = (
            f"{int(time.time() * 1000)}_"
            f"{index}_"
            f"{Path(original_filename).name}"
        )

        input_path = (
            UPLOADS /
            safe_filename
        )

        # ----------------------------------------------------
        # SAVE IMAGE
        # ----------------------------------------------------

        try:

            contents = (
                await upload.read()
            )

            if not contents:

                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Image is empty: "
                        f"{original_filename}"
                    ),
                )

            input_path.write_bytes(
                contents
            )

        except HTTPException:
            raise

        except Exception as exc:

            print()
            print("=" * 80)
            print("UPLOAD ERROR")
            print("=" * 80)
            traceback.print_exc()
            print("=" * 80)

            raise HTTPException(
                status_code=500,
                detail=(
                    f"Could not save "
                    f"{original_filename}: "
                    f"{exc}"
                ),
            )

        # ----------------------------------------------------
        # YOLO
        # ----------------------------------------------------

        try:

            result = analyze_image(
                str(input_path),
                confidence_threshold,
            )

        except Exception as exc:

            print()
            print("=" * 80)
            print("!!! YOLO ERROR !!!")
            print(
                f"IMAGE: {original_filename}"
            )
            print("=" * 80)

            traceback.print_exc()

            print("=" * 80)

            raise HTTPException(
                status_code=500,
                detail=(
                    f"YOLO processing failed "
                    f"for {original_filename}: "
                    f"{exc}"
                ),
            )

        # ----------------------------------------------------
        # DETECTIONS FROM YOLO
        # ----------------------------------------------------

        image_detections = []

        for detection in result.get(
            "detections",
            [],
        ):

            item = {

                "target_class":
                    detection.get(
                        "type",
                        "unknown",
                    ),

                "confidence_score":
                    float(
                        detection.get(
                            "confidence",
                            0.0,
                        )
                    ),

                "bounding_box":
                    list(
                        detection.get(
                            "bbox",
                            [],
                        )
                    ),

                "source_image":
                    original_filename,

                "crop_url":
                    detection.get(
                        "crop_url",
                        "",
                    ),
            }

            image_detections.append(
                item
            )

            all_detections.append(
                item
            )

        # ----------------------------------------------------
        # IMAGE RECORD
        # ----------------------------------------------------

        image_records.append({

            "image_id":
                original_filename,

            "raw_size_bytes":
                int(
                    result.get(
                        "raw_size_bytes",
                        input_path.stat().st_size,
                    )
                ),

            "width":
                int(
                    result.get(
                        "image_width",
                        0,
                    )
                ),

            "height":
                int(
                    result.get(
                        "image_height",
                        0,
                    )
                ),

            "annotated_image_url":
                result.get(
                    "annotated_image_url",
                    "",
                ),

            "detections":
                image_detections,

            "sent":
                [],

            "discarded":
                [],
        })

    # ========================================================
    # MISSION DECISION
    # ========================================================

    try:

        decision_result = make_decisions(

            detections=
                all_detections,

            bandwidth_kb=
                bandwidth_kb,

            mission_mode=
                mission_mode,
        )

    except Exception as exc:

        print()
        print("=" * 80)
        print("!!! MISSION DECISION ERROR !!!")
        print("=" * 80)

        traceback.print_exc()

        print("=" * 80)

        raise HTTPException(
            status_code=500,
            detail=(
                f"Mission decision failed: "
                f"{exc}"
            ),
        )

    sent = decision_result[
        "sent"
    ]

    discarded = decision_result[
        "discarded"
    ]

    # ========================================================
    # SPLIT DECISIONS BY IMAGE
    # ========================================================

    sent_by_image = {}
    discarded_by_image = {}

    for item in sent:

        source = item.get(
            "source_image",
            "",
        )

        sent_by_image.setdefault(
            source,
            [],
        ).append(
            item
        )

    for item in discarded:

        source = item.get(
            "source_image",
            "",
        )

        discarded_by_image.setdefault(
            source,
            [],
        ).append(
            item
        )

    for record in image_records:

        image_id = record[
            "image_id"
        ]

        record["sent"] = (
            sent_by_image.get(
                image_id,
                [],
            )
        )

        record["discarded"] = (
            discarded_by_image.get(
                image_id,
                [],
            )
        )

    # ========================================================
    # RAW / OPTIMIZED DATA
    # ========================================================

    raw_bytes = sum(

        int(
            image[
                "raw_size_bytes"
            ]
        )

        for image
        in image_records
    )

    raw_kb = (
        raw_bytes /
        1024.0
    )

    optimized_kb = sum(

        float(
            item.get(
                "data_cost_kb",
                0,
            )
        )

        for item
        in sent
    )

    optimized_bytes = int(
        round(
            optimized_kb *
            1024.0
        )
    )

    # ========================================================
    # TELEMETRY / DOWNLINK
    # ========================================================

    telemetry = telemetry_dict()

    downlink_kbps = float(
        telemetry.get(
            "bandwidth_available_kbps",
            500.0,
        )
    )

    if downlink_kbps > 0:

        raw_transmission_ms = (
            raw_bytes *
            8.0 /
            downlink_kbps
        )

        optimized_transmission_ms = (
            optimized_bytes *
            8.0 /
            downlink_kbps
        )

    else:

        raw_transmission_ms = 0.0
        optimized_transmission_ms = 0.0

    # ========================================================
    # REDUCTION
    # ========================================================

    if raw_bytes > 0:

        bandwidth_reduction = (

            (
                raw_bytes -
                optimized_bytes
            )
            /
            raw_bytes
            *
            100.0
        )

    else:

        bandwidth_reduction = 0.0

    bandwidth_reduction = max(
        0.0,
        min(
            100.0,
            bandwidth_reduction,
        )
    )

    # ========================================================
    # LATENCY REDUCTION
    # ========================================================

    if raw_transmission_ms > 0:

        latency_reduction = (

            (
                raw_transmission_ms -
                optimized_transmission_ms
            )
            /
            raw_transmission_ms
            *
            100.0
        )

    else:

        latency_reduction = 0.0

    # ========================================================
    # ENERGY
    # ========================================================

    raw_energy = (
        energy_estimate(
            raw_bytes
        )
    )

    optimized_energy = (
        energy_estimate(
            optimized_bytes
        )
    )

    if raw_energy > 0:

        energy_reduction = (

            (
                raw_energy -
                optimized_energy
            )
            /
            raw_energy
            *
            100.0
        )

    else:

        energy_reduction = 0.0

    # ========================================================
    # PACKET BREAKDOWN
    # ========================================================

    packet_breakdown = {}

    for item in sent:

        package = item.get(
            "data_package",
            "UNKNOWN",
        )

        cost = float(
            item.get(
                "data_cost_kb",
                0,
            )
        )

        packet_breakdown[
            package
        ] = (
            packet_breakdown.get(
                package,
                0.0,
            )
            +
            cost
        )

    # ========================================================
    # PROCESSING TIME
    # ========================================================

    inference_ms = (
        time.perf_counter()
        -
        started
    ) * 1000.0

    # ========================================================
    # UPDATE HISTORY
    # ========================================================

    history_item = {

        "time":
            time.strftime(
                "%H:%M:%S"
            ),

        "mission_mode":
            mission_mode,

        "detections":
            len(all_detections),

        "transmitted":
            len(sent),

        "discarded":
            len(discarded),

        "raw_kb":
            round(
                raw_kb,
                2,
            ),

        "optimized_kb":
            round(
                optimized_kb,
                2,
            ),

        "reduction":
            round(
                bandwidth_reduction,
                2,
            ),
    }

    HISTORY.append(
        history_item
    )

    if len(HISTORY) > 30:

        del HISTORY[
            :-30
        ]

    # ========================================================
    # SAVE OUTPUT
    # ========================================================

    (
        RUNTIME /
        "final_output.json"
    ).write_text(

        json.dumps(
            sent,
            indent=2,
        ),

        encoding="utf-8",
    )

    # ========================================================
    # FINAL RESPONSE
    # ========================================================

    return {

        "summary": {

            "mission_mode":
                mission_mode,

            "bandwidth_budget_kb":
                bandwidth_kb,

            "images_processed":
                len(image_records),

            "total_detections":
                len(all_detections),

            "transmitted":
                len(sent),

            "discarded":
                len(discarded),

            "raw_bytes":
                raw_bytes,

            "optimized_bytes":
                optimized_bytes,

            "raw_kb":
                round(
                    raw_kb,
                    2,
                ),

            "optimized_kb":
                round(
                    optimized_kb,
                    2,
                ),

            "bandwidth_reduction_percent":
                round(
                    bandwidth_reduction,
                    2,
                ),

            "raw_transmission_ms":
                round(
                    raw_transmission_ms,
                    2,
                ),

            "optimized_transmission_ms":
                round(
                    optimized_transmission_ms,
                    2,
                ),

            "latency_reduction_percent":
                round(
                    latency_reduction,
                    2,
                ),

            "raw_energy_j":
                round(
                    raw_energy,
                    3,
                ),

            "optimized_energy_j":
                round(
                    optimized_energy,
                    3,
                ),

            "energy_reduction_percent":
                round(
                    energy_reduction,
                    2,
                ),

            "inference_ms":
                round(
                    inference_ms,
                    2,
                ),

            "packet_kb_by_type":
                packet_breakdown,

            "battery_percent":
                float(
                    telemetry.get(
                        "battery_percent",
                        78.0,
                    )
                ),

            "cpu_load_percent":
                float(
                    telemetry.get(
                        "cpu_load_percent",
                        10.0,
                    )
                ),

            "ground_link_active":
                bool(
                    telemetry.get(
                        "ground_link_active",
                        True,
                    )
                ),
        },

        "images":
            image_records,

        "sent":
            sent,

        "discarded":
            discarded,
    }


# ============================================================
# DASHBOARD
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse,
)
def dashboard():

    return r"""
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
>

<title>
    Orbital Edge Intelligence
</title>

<style>

:root {

    --bg:#050505;
    --panel:#0b0b0b;
    --panel2:#11110f;
    --line:#2b2413;
    --gold:#e5b94f;
    --gold2:#f4d276;
    --white:#f2eee5;
    --muted:#8b877e;
    --green:#42dc91;
    --red:#ff5b5b;

}

* {
    box-sizing:border-box;
}

body {

    margin:0;

    min-height:100vh;

    background:
        radial-gradient(
            circle at 50% -20%,
            #211909 0%,
            #090909 35%,
            #050505 75%
        );

    color:var(--white);

    font-family:
        Arial,
        Helvetica,
        sans-serif;

}

.container {

    width:min(
        1500px,
        94%
    );

    margin:auto;

    padding:
        25px 0 60px;

}

.header {

    display:flex;

    justify-content:
        space-between;

    align-items:center;

    gap:20px;

    padding-bottom:20px;

    border-bottom:
        1px solid var(--line);

}

.brand {

    display:flex;

    gap:15px;

    align-items:center;

}

.logo {

    width:55px;

    height:55px;

    border:
        1px solid var(--gold);

    border-radius:50%;

    display:grid;

    place-items:center;

    color:var(--gold2);

    font-weight:900;

}

h1 {

    margin:0;

    color:var(--gold2);

    font-size:
        clamp(
            24px,
            3vw,
            42px
        );

    letter-spacing:3px;

}

.subtitle {

    color:var(--muted);

    font-size:11px;

    letter-spacing:1px;

    margin-top:6px;

}

.actions {

    display:flex;

    align-items:center;

    gap:10px;

    flex-wrap:wrap;

}

button {

    background:#10100e;

    color:var(--gold2);

    border:
        1px solid #3a3018;

    border-radius:8px;

    padding:
        11px 15px;

    cursor:pointer;

    font-weight:700;

}

button:hover {

    border-color:
        var(--gold);

}

.status {

    border:
        1px solid rgba(
            66,
            220,
            145,
            .35
        );

    color:var(--green);

    padding:
        9px 12px;

    border-radius:20px;

    font-size:10px;

}

.metrics {

    display:grid;

    grid-template-columns:
        repeat(
            7,
            minmax(
                0,
                1fr
            )
        );

    gap:12px;

    margin:
        15px 0;

}

.card {

    background:
        linear-gradient(
            145deg,
            #11110f,
            #090909
        );

    border:
        1px solid var(--line);

    border-radius:12px;

    padding:17px;

}

.label {

    color:var(--muted);

    font-size:10px;

    letter-spacing:1.6px;

    text-transform:uppercase;

}

.value {

    margin-top:8px;

    font-size:27px;

    font-weight:900;

}

.gold {
    color:var(--gold2);
}

.progress {

    height:5px;

    margin-top:10px;

    background:#1d1b17;

    overflow:hidden;

    border-radius:10px;

}

.progress div {

    height:100%;

    background:var(--gold);

}

.layout {

    display:grid;

    grid-template-columns:
        285px
        minmax(
            0,
            1fr
        )
        300px;

    gap:14px;

}

.title {

    color:var(--gold);

    font-size:13px;

    letter-spacing:2px;

    text-transform:uppercase;

    margin-bottom:16px;

}

label {

    display:block;

    color:#a9a399;

    font-size:11px;

    margin-bottom:7px;

}

select {

    width:100%;

    color:var(--white);

    background:#070707;

    border:
        1px solid #342914;

    border-radius:7px;

    padding:11px;

    margin-bottom:18px;

}

input[type=range] {

    width:100%;

    accent-color:var(--gold);

    margin-bottom:18px;

}

.filebox {

    padding:17px;

    text-align:center;

    border:
        1px dashed #604a18;

    border-radius:10px;

}

.filebox strong {

    display:block;

    color:var(--gold2);

    margin-bottom:8px;

}

.filebox input {

    width:100%;

    font-size:10px;

}

.primary {

    width:100%;

    margin-top:14px;

    padding:14px;

    background:var(--gold);

    color:#080808;

    border:none;

    font-size:13px;

    font-weight:900;

}

.analysis {

    min-width:0;

}

.image-grid {

    display:grid;

    grid-template-columns:
        repeat(
            auto-fit,
            minmax(
                300px,
                1fr
            )
        );

    gap:12px;

}

.image-card {

    overflow:hidden;

    background:#070707;

    border:
        1px solid #292216;

    border-radius:11px;

}

.image-card img {

    width:100%;

    max-height:320px;

    object-fit:contain;

    display:block;

    background:#020202;

}

.image-info {

    padding:13px;

}

.filename {

    color:var(--gold2);

    font-size:12px;

    font-weight:800;

    word-break:break-all;

}

.small {

    color:var(--muted);

    font-size:10px;

    line-height:1.6;

}

.detection {

    display:flex;

    justify-content:space-between;

    align-items:center;

    gap:12px;

    padding:10px 0;

    border-bottom:
        1px solid #1d1b18;

}

.conf {

    color:var(--gold2);

    font-weight:900;

}

.pill {

    display:inline-block;

    margin-top:5px;

    padding:4px 7px;

    border-radius:4px;

    font-size:8px;

    font-weight:900;

    letter-spacing:1px;

}

.transmit {

    color:var(--green);

    border:
        1px solid rgba(
            66,
            220,
            145,
            .35
        );

}

.discard {

    color:var(--red);

    border:
        1px solid rgba(
            255,
            91,
            91,
            .35
        );

}

.stats {

    display:grid;

    gap:8px;

}

.stat {

    display:flex;

    justify-content:space-between;

    gap:10px;

    padding:
        8px 0;

    border-bottom:
        1px solid #1d1b18;

    color:#a8a39a;

    font-size:11px;

}

.stat b {

    color:var(--white);

}

.compare {

    display:grid;

    grid-template-columns:
        1fr
        1fr;

    gap:12px;

}

.big {

    margin:
        8px 0;

    font-size:30px;

    font-weight:900;

}

.bar {

    height:10px;

    margin-top:13px;

    border-radius:10px;

    overflow:hidden;

    background:#1a1916;

}

.bar div {

    height:100%;

}

.raw {
    background:#77736a;
}

.edge {
    background:var(--gold);
}

.chart {

    height:150px;

    display:flex;

    align-items:flex-end;

    gap:7px;

    border-left:
        1px solid #28251f;

    border-bottom:
        1px solid #28251f;

    padding:
        10px 5px 0;

}

.chart-item {

    position:relative;

    flex:1;

    min-width:5px;

    max-width:45px;

    background:var(--gold);

    border-radius:
        3px 3px 0 0;

}

.chart-item span {

    position:absolute;

    bottom:-17px;

    left:0;

    right:0;

    text-align:center;

    color:#66625c;

    font-size:8px;

}

.empty {

    min-height:220px;

    display:grid;

    place-items:center;

    color:#5e5a53;

    border:
        1px dashed #29261f;

    border-radius:10px;

    text-align:center;

}

.error {

    color:var(--red);

    border:
        1px solid rgba(
            255,
            91,
            91,
            .35
        );

    padding:12px;

    border-radius:7px;

}

@media(max-width:1150px) {

    .metrics {

        grid-template-columns:
            repeat(
                3,
                1fr
            );

    }

    .layout {

        grid-template-columns:
            1fr;

    }

}

@media(max-width:650px) {

    .metrics {

        grid-template-columns:
            repeat(
                2,
                1fr
            );

    }

    .compare {

        grid-template-columns:
            1fr;

    }

}

</style>

</head>

<body>

<div class="container">


<header class="header">

    <div class="brand">

        <div class="logo">
            OE
        </div>

        <div>

            <h1>
                ORBITAL EDGE INTELLIGENCE
            </h1>

            <div class="subtitle">
                MISSION CONTROL • ONBOARD VISION • AUTONOMOUS DOWNLINK OPTIMISATION
            </div>

        </div>

    </div>


    <div class="actions">

        <button
            onclick="triggerOrbit()"
        >
            TRIGGER ORBIT PASS
        </button>

        <button
            onclick="resetSystem()"
        >
            RESET
        </button>

        <div
            id="link"
            class="status"
        >
            ● GROUND LINK ACTIVE
        </div>

    </div>

</header>


<!-- =======================================================
     METRICS
======================================================= -->

<div class="metrics">

    <div class="card">

        <div class="label">
            Battery
        </div>

        <div
            id="battery"
            class="value"
        >
            78.0%
        </div>

        <div class="progress">
            <div
                id="batteryBar"
                style="width:78%"
            ></div>
        </div>

    </div>


    <div class="card">

        <div class="label">
            Edge CPU
        </div>

        <div
            id="cpu"
            class="value"
        >
            10.0%
        </div>

        <div class="progress">
            <div
                id="cpuBar"
                style="width:10%"
            ></div>
        </div>

    </div>


    <div class="card">

        <div class="label">
            RF Downlink
        </div>

        <div
            id="bandwidth"
            class="value gold"
        >
            500
        </div>

        <div class="small">
            KBPS AVAILABLE
        </div>

    </div>


    <div class="card">

        <div class="label">
            Raw Ingested
        </div>

        <div
            id="rawTop"
            class="value"
        >
            0 KB
        </div>

    </div>


    <div class="card">

        <div class="label">
            Payload Sent
        </div>

        <div
            id="payloadTop"
            class="value gold"
        >
            0 KB
        </div>

    </div>


    <div class="card">

        <div class="label">
            Data Reduction
        </div>

        <div
            id="reductionTop"
            class="value gold"
        >
            0%
        </div>

    </div>


    <div class="card">

        <div class="label">
            Mission State
        </div>

        <div
            id="missionState"
            class="value"
            style="font-size:18px"
        >
            READY
        </div>

    </div>

</div>


<div class="layout">


<!-- =======================================================
     LEFT
======================================================= -->

<div class="card">

    <div class="title">
        Mission Configuration
    </div>


    <label>
        Mission Mode
    </label>

    <select id="missionMode">

        <option value="MARITIME_SURVEILLANCE">
            Maritime Surveillance
        </option>

        <option value="URBAN_MONITORING">
            Urban Monitoring
        </option>

        <option value="DISASTER_RESPONSE">
            Disaster Response
        </option>

        <option value="WILDFIRE_RESPONSE">
            Wildfire Response
        </option>

        <option value="GENERAL">
            General Observation
        </option>

    </select>


    <label>
        Detection Threshold:
        <span id="thresholdText">
            0.25
        </span>
    </label>

    <input
        id="threshold"
        type="range"
        min="0.01"
        max="0.99"
        step="0.01"
        value="0.25"
        oninput="
            document.getElementById(
                'thresholdText'
            ).innerText =
            Number(this.value).toFixed(2)
        "
    >


    <label>
        Shared Pass Budget:
        <span id="budgetText">
            20 KB
        </span>
    </label>

    <input
        id="budget"
        type="range"
        min="1"
        max="100"
        value="20"
        oninput="
            document.getElementById(
                'budgetText'
            ).innerText =
            this.value + ' KB'
        "
    >


    <div class="filebox">

        <strong>
            SATELLITE FRAME INGESTION
        </strong>

        <div class="small">
            Select one or multiple observation frames.
            All frames share one orbital downlink budget.
        </div>

        <br>

        <input
            id="files"
            type="file"
            accept=".jpg,.jpeg,.png"
            multiple
        >

    </div>


    <button
        class="primary"
        onclick="runPipeline()"
    >
        RUN ONBOARD PIPELINE
    </button>


    <div
        id="logs"
        class="small"
        style="
            margin-top:14px;
            max-height:180px;
            overflow:auto;
            font-family:monospace;
        "
    >
        [SYSTEM] READY
    </div>

</div>


<!-- =======================================================
     CENTER
======================================================= -->

<div class="analysis">


    <div class="card">

        <div class="title">
            Live Mission Analysis
        </div>

        <div
            id="analysisStatus"
            class="small"
        >
            Select satellite imagery and run the onboard pipeline.
        </div>


        <div
            id="imageGrid"
            class="image-grid"
            style="margin-top:15px;"
        >

            <div class="empty">
                DETECTION CHANNEL STANDBY
            </div>

        </div>

    </div>


    <div
        class="card"
        style="margin-top:14px;"
    >

        <div class="title">
            Conventional Downlink vs Edge-AI
        </div>


        <div class="compare">


            <div class="card">

                <div class="label">
                    BEFORE — RAW IMAGERY
                </div>

                <div
                    id="before"
                    class="big"
                >
                    0 KB
                </div>

                <div class="small">
                    Complete raster imagery sent to Earth.
                </div>

                <div class="stat">

                    <span>
                        Transmission
                    </span>

                    <b id="beforeTime">
                        0 ms
                    </b>

                </div>

                <div class="stat">

                    <span>
                        Energy model
                    </span>

                    <b id="beforeEnergy">
                        0 J
                    </b>

                </div>

                <div class="bar">
                    <div
                        class="raw"
                        style="width:100%"
                    ></div>
                </div>

            </div>


            <div class="card">

                <div class="label">
                    AFTER — EDGE PAYLOAD
                </div>

                <div
                    id="after"
                    class="big gold"
                >
                    0 KB
                </div>

                <div class="small">
                    Only mission-relevant evidence is sent.
                </div>

                <div class="stat">

                    <span>
                        Transmission
                    </span>

                    <b id="afterTime">
                        0 ms
                    </b>

                </div>

                <div class="stat">

                    <span>
                        Energy model
                    </span>

                    <b id="afterEnergy">
                        0 J
                    </b>

                </div>

                <div class="bar">

                    <div
                        id="edgeBar"
                        class="edge"
                        style="width:0%"
                    ></div>

                </div>

            </div>

        </div>


        <div class="stat"
             style="margin-top:20px">

            <span>
                DOWNLINK SAVED
            </span>

            <b
                id="saved"
                class="gold"
            >
                0%
            </b>

        </div>

        <div class="bar">

            <div
                id="savedBar"
                class="edge"
                style="width:0%"
            ></div>

        </div>

    </div>


    <div
        class="card"
        style="margin-top:14px"
    >

        <div class="title">
            Efficiency History
        </div>

        <div
            id="chart"
            class="chart"
        ></div>

    </div>

</div>


<!-- =======================================================
     RIGHT
======================================================= -->

<div>


    <div class="card">

        <div class="title">
            Mission Telemetry
        </div>


        <div class="stat">

            <span>
                Detections
            </span>

            <b id="detections">
                0
            </b>

        </div>


        <div class="stat">

            <span>
                Transmitted
            </span>

            <b
                id="transmitted"
                style="color:var(--green)"
            >
                0
            </b>

        </div>


        <div class="stat">

            <span>
                Discarded
            </span>

            <b
                id="discarded"
                style="color:var(--red)"
            >
                0
            </b>

        </div>


        <div class="stat">

            <span>
                Inference
            </span>

            <b id="inference">
                0 ms
            </b>

        </div>


        <div class="stat">

            <span>
                Latency saved
            </span>

            <b
                id="latency"
                class="gold"
            >
                0%
            </b>

        </div>


        <div class="stat">

            <span>
                Energy saved
            </span>

            <b
                id="energy"
                class="gold"
            >
                0%
            </b>

        </div>

    </div>


    <div
        class="card"
        style="margin-top:14px"
    >

        <div class="title">
            Decision Breakdown
        </div>

        <div
            id="breakdown"
            class="stats"
        >

            <div class="small">
                Awaiting analysis.
            </div>

        </div>

    </div>


    <div
        class="card"
        style="margin-top:14px"
    >

        <div class="title">
            Autonomous Edge Chain
        </div>

        <div
            class="small"
            style="line-height:2"
        >

            IMAGE INGESTION
            <br>

            <span class="gold">↓</span>
            <br>

            YOLOv8n
            <br>

            <span class="gold">↓</span>
            <br>

            CONFIDENCE
            <br>

            <span class="gold">↓</span>
            <br>

            MISSION RELEVANCE
            <br>

            <span class="gold">↓</span>
            <br>

            PRIORITY
            <br>

            <span class="gold">↓</span>
            <br>

            SHARED BANDWIDTH
            <br>

            <span class="gold">↓</span>
            <br>

            TRANSMIT / DISCARD

        </div>

    </div>

</div>

</div>

</div>


<script>


// ==========================================================
// HELPERS
// ==========================================================

function kb(bytes) {

    return (
        Number(bytes || 0)
        /
        1024
    );

}


function formatKB(bytes) {

    return (
        kb(bytes)
        .toFixed(2)
        +
        " KB"
    );

}


function esc(value) {

    return String(
        value
    )
    .replace(
        /&/g,
        "&amp;"
    )
    .replace(
        /</g,
        "&lt;"
    )
    .replace(
        />/g,
        "&gt;"
    )
    .replace(
        /"/g,
        "&quot;"
    )
    .replace(
        /'/g,
        "&#039;"
    );

}


function log(message) {

    const el =
        document.getElementById(
            "logs"
        );

    const t =
        new Date()
        .toLocaleTimeString();

    el.innerHTML +=
        "<br>[" +
        t +
        "] " +
        esc(message);

    el.scrollTop =
        el.scrollHeight;

}


// ==========================================================
// TELEMETRY
// ==========================================================

async function refreshTelemetry() {

    try {

        const response =
            await fetch(
                "/satellite/status"
            );

        if (!response.ok)
            return;

        const data =
            await response.json();


        const battery =
            Number(
                data.battery_percent
            );


        const cpu =
            Number(
                data.cpu_load_percent
            );


        document.getElementById(
            "battery"
        ).innerText =
            battery.toFixed(1)
            + "%";


        document.getElementById(
            "batteryBar"
        ).style.width =
            Math.max(
                0,
                Math.min(
                    100,
                    battery
                )
            )
            + "%";


        document.getElementById(
            "cpu"
        ).innerText =
            cpu.toFixed(1)
            + "%";


        document.getElementById(
            "cpuBar"
        ).style.width =
            Math.max(
                0,
                Math.min(
                    100,
                    cpu
                )
            )
            + "%";


        document.getElementById(
            "bandwidth"
        ).innerText =
            Number(
                data.bandwidth_available_kbps
            ).toFixed(0);


        document.getElementById(
            "rawTop"
        ).innerText =
            formatKB(
                data.raw_data_size_bytes
            );


        document.getElementById(
            "payloadTop"
        ).innerText =
            formatKB(
                data.metadata_size_bytes
            );


        const active =
            Boolean(
                data.ground_link_active
            );


        const link =
            document.getElementById(
                "link"
            );


        link.innerText =
            active
                ? "● GROUND LINK ACTIVE"
                : "● GROUND LINK OFFLINE";


        link.style.color =
            active
                ? "var(--green)"
                : "var(--red)";

    }

    catch(error) {

        console.error(
            error
        );

    }

}


// ==========================================================
// MAIN PIPELINE
// ==========================================================

async function runPipeline() {

    const input =
        document.getElementById(
            "files"
        );


    if (
        !input.files.length
    ) {

        alert(
            "Select at least one image."
        );

        return;

    }


    const mode =
        document.getElementById(
            "missionMode"
        ).value;


    const threshold =
        Number(
            document.getElementById(
                "threshold"
            ).value
        );


    const budget =
        Number(
            document.getElementById(
                "budget"
            ).value
        );


    const form =
        new FormData();


    for (
        const file
        of input.files
    ) {

        form.append(
            "files",
            file
        );

    }


    form.append(
        "mission_mode",
        mode
    );


    form.append(
        "bandwidth_kb",
        budget
    );


    form.append(
        "confidence_threshold",
        threshold
    );


    document.getElementById(
        "missionState"
    ).innerText =
        "PROCESSING";


    document.getElementById(
        "analysisStatus"
    ).innerText =
        "Running YOLOv8n and mission-aware downlink selection...";


    log(
        "Starting onboard pipeline."
    );


    log(
        "Mission: " +
        mode
    );


    log(
        "Frames: " +
        input.files.length
    );


    log(
        "Pass budget: " +
        budget +
        " KB"
    );


    try {

        const response =
            await fetch(
                "/api/analyze",
                {
                    method:
                        "POST",

                    body:
                        form,
                }
            );


        const text =
            await response.text();


        let result;


        try {

            result =
                JSON.parse(
                    text
                );

        }

        catch {

            throw new Error(
                "Backend returned: " +
                text.substring(
                    0,
                    300
                )
            );

        }


        if (!response.ok) {

            throw new Error(
                result.detail
                ||
                "Analysis failed."
            );

        }


        render(
            result
        );


        document.getElementById(
            "missionState"
        ).innerText =
            "MISSION COMPLETE";


        document.getElementById(
            "analysisStatus"
        ).innerText =
            "Onboard analysis complete.";


        log(
            "Analysis complete."
        );


        log(
            "Detections: " +
            result.summary.total_detections
        );


        log(
            "Transmitted: " +
            result.summary.transmitted
        );


        log(
            "Discarded: " +
            result.summary.discarded
        );


        log(
            "Reduction: " +
            result.summary.bandwidth_reduction_percent +
            "%"
        );


        await refreshTelemetry();

        await refreshHistory();

    }

    catch(error) {

        console.error(
            error
        );


        document.getElementById(
            "missionState"
        ).innerText =
            "ERROR";


        document.getElementById(
            "analysisStatus"
        ).innerHTML =
            "<div class='error'>" +
            esc(
                error.message
            ) +
            "</div>";


        log(
            "ERROR: " +
            error.message
        );

    }

}


// ==========================================================
// RENDER RESULT
// ==========================================================

function render(
    result
) {

    const s =
        result.summary;


    document.getElementById(
        "detections"
    ).innerText =
        s.total_detections;


    document.getElementById(
        "transmitted"
    ).innerText =
        s.transmitted;


    document.getElementById(
        "discarded"
    ).innerText =
        s.discarded;


    document.getElementById(
        "inference"
    ).innerText =
        Number(
            s.inference_ms
        ).toFixed(1)
        +
        " ms";


    document.getElementById(
        "latency"
    ).innerText =
        Number(
            s.latency_reduction_percent
        ).toFixed(1)
        +
        "%";


    document.getElementById(
        "energy"
    ).innerText =
        Number(
            s.energy_reduction_percent
        ).toFixed(1)
        +
        "%";


    document.getElementById(
        "reductionTop"
    ).innerText =
        Number(
            s.bandwidth_reduction_percent
        ).toFixed(1)
        +
        "%";


    document.getElementById(
        "before"
    ).innerText =
        formatKB(
            s.raw_bytes
        );


    document.getElementById(
        "after"
    ).innerText =
        formatKB(
            s.optimized_bytes
        );


    document.getElementById(
        "beforeTime"
    ).innerText =
        Number(
            s.raw_transmission_ms
        ).toFixed(1)
        +
        " ms";


    document.getElementById(
        "afterTime"
    ).innerText =
        Number(
            s.optimized_transmission_ms
        ).toFixed(1)
        +
        " ms";


    document.getElementById(
        "beforeEnergy"
    ).innerText =
        Number(
            s.raw_energy_j
        ).toFixed(3)
        +
        " J";


    document.getElementById(
        "afterEnergy"
    ).innerText =
        Number(
            s.optimized_energy_j
        ).toFixed(3)
        +
        " J";


    document.getElementById(
        "saved"
    ).innerText =
        Number(
            s.bandwidth_reduction_percent
        ).toFixed(1)
        +
        "%";


    document.getElementById(
        "savedBar"
    ).style.width =
        Math.max(
            0,
            Math.min(
                100,
                s.bandwidth_reduction_percent
            )
        )
        +
        "%";


    const payloadPercent =
        s.raw_bytes > 0
            ?
            (
                s.optimized_bytes /
                s.raw_bytes *
                100
            )
            :
            0;


    document.getElementById(
        "edgeBar"
    ).style.width =
        Math.max(
            0,
            Math.min(
                100,
                payloadPercent
            )
        )
        +
        "%";


    renderImages(
        result.images
    );


    renderBreakdown(
        result
    );

}


// ==========================================================
// IMAGE RESULTS
// ==========================================================

function renderImages(
    images
) {

    const grid =
        document.getElementById(
            "imageGrid"
        );


    grid.innerHTML =
        "";


    images.forEach(
        image => {

            const card =
                document.createElement(
                    "div"
                );


            card.className =
                "image-card";


            const img =
                document.createElement(
                    "img"
                );


            img.src =
                image.annotated_image_url;


            img.alt =
                image.image_id;


            const info =
                document.createElement(
                    "div"
                );


            info.className =
                "image-info";


            let html = "";


            html +=
                "<div class='filename'>" +
                esc(
                    image.image_id
                ) +
                "</div>";


            html +=
                "<div class='small'>" +
                image.width +
                " × " +
                image.height +
                " • " +
                formatKB(
                    image.raw_size_bytes
                ) +
                "</div>";


            for (
                const d
                of image.detections
            ) {

                const isSent =
                    image.sent.some(
                        item =>
                            JSON.stringify(
                                item.bounding_box
                            )
                            ===
                            JSON.stringify(
                                d.bounding_box
                            )
                    );


                const decision =
                    isSent
                    ? "TRANSMIT"
                    : "DISCARD";


                const cls =
                    isSent
                    ? "transmit"
                    : "discard";


                html +=

                    "<div class='detection'>" +

                        "<div>" +

                            "<div>" +

                                "<strong>" +
                                esc(
                                    d.target_class
                                ) +
                                "</strong>" +

                            "</div>" +

                            "<div class='small'>" +
                            "BOX [" +
                            d.bounding_box.join(
                                ", "
                            ) +
                            "]" +
                            "</div>" +

                        "</div>" +

                        "<div style='text-align:right'>" +

                            "<div class='conf'>" +
                            (
                                Number(
                                    d.confidence_score
                                )
                                *
                                100
                            ).toFixed(1) +
                            "%" +
                            "</div>" +

                            "<span class='pill " +
                            cls +
                            "'>" +
                            decision +
                            "</span>" +

                        "</div>" +

                    "</div>";

            }


            info.innerHTML =
                html;


            card.appendChild(
                img
            );

            card.appendChild(
                info
            );

            grid.appendChild(
                card
            );

        }
    );

}


// ==========================================================
// BREAKDOWN
// ==========================================================

function renderBreakdown(
    result
) {

    const box =
        document.getElementById(
            "breakdown"
        );


    const total =
        result.summary.total_detections;


    const sent =
        result.summary.transmitted;


    const discarded =
        result.summary.discarded;


    const sentPct =
        total > 0
        ?
        sent /
        total *
        100
        :
        0;


    const discardedPct =
        total > 0
        ?
        discarded /
        total *
        100
        :
        0;


    box.innerHTML =

        "<div class='stat'>" +

            "<span>TRANSMIT</span>" +

            "<b style='color:var(--green)'>" +
            sent +
            " (" +
            sentPct.toFixed(1) +
            "%)" +
            "</b>" +

        "</div>" +


        "<div class='stat'>" +

            "<span>DISCARD</span>" +

            "<b style='color:var(--red)'>" +
            discarded +
            " (" +
            discardedPct.toFixed(1) +
            "%)" +
            "</b>" +

        "</div>" +


        "<div class='stat'>" +

            "<span>MISSION</span>" +

            "<b>" +
            esc(
                result.summary.mission_mode
            ) +
            "</b>" +

        "</div>" +


        "<div class='stat'>" +

            "<span>PASS BUDGET</span>" +

            "<b>" +
            result.summary.bandwidth_budget_kb +
            " KB</b>" +

        "</div>";

}


// ==========================================================
// HISTORY GRAPH
// ==========================================================

async function refreshHistory() {

    try {

        const response =
            await fetch(
                "/api/history"
            );


        if (!response.ok)
            return;


        const data =
            await response.json();


        const history =
            data.history || [];


        const chart =
            document.getElementById(
                "chart"
            );


        chart.innerHTML =
            "";


        if (!history.length)
            return;


        const recent =
            history.slice(
                -15
            );


        recent.forEach(
            item => {

                const bar =
                    document.createElement(
                        "div"
                    );


                bar.className =
                    "chart-item";


                const value =
                    Number(
                        item.reduction
                    );


                bar.style.height =
                    Math.max(
                        3,
                        value
                    )
                    +
                    "%";


                bar.title =
                    item.mission_mode +
                    " • " +
                    value +
                    "% reduction";


                const label =
                    document.createElement(
                        "span"
                    );


                label.innerText =
                    item.time;


                bar.appendChild(
                    label
                );


                chart.appendChild(
                    bar
                );

            }
        );

    }

    catch(error) {

        console.error(
            error
        );

    }

}


// ==========================================================
// ORBIT
// ==========================================================

async function triggerOrbit() {

    log(
        "Triggering simulated orbit pass..."
    );


    try {

        const response =
            await fetch(
                "/simulation/start",
                {
                    method:
                        "POST"
                }
            );


        const result =
            await response.json();


        log(
            result.message
            ||
            "Orbit pass triggered."
        );


        await refreshTelemetry();

    }

    catch(error) {

        log(
            "Orbit pass failed."
        );

    }

}


// ==========================================================
// RESET
// ==========================================================

async function resetSystem() {

    try {

        await fetch(
            "/simulation/reset",
            {
                method:
                    "POST"
            }
        );


        document.getElementById(
            "missionState"
        ).innerText =
            "READY";


        document.getElementById(
            "analysisStatus"
        ).innerText =
            "System reset. Ready for new imagery.";


        document.getElementById(
            "imageGrid"
        ).innerHTML =
            '<div class="empty">DETECTION CHANNEL STANDBY</div>';


        document.getElementById(
            "breakdown"
        ).innerHTML =
            '<div class="small">Awaiting analysis.</div>';


        document.getElementById(
            "reductionTop"
        ).innerText =
            "0%";


        document.getElementById(
            "saved"
        ).innerText =
            "0%";


        log(
            "System reset."
        );


        await refreshTelemetry();

        await refreshHistory();

    }

    catch(error) {

        log(
            "Reset failed."
        );

    }

}


// ==========================================================
// START
// ==========================================================

refreshTelemetry();

refreshHistory();

setInterval(
    refreshTelemetry,
    2000
);

</script>

</body>

</html>
"""


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )