# ============================================================
# ORBITAL EDGE INTELLIGENCE
# mission_engine.py
# ============================================================

CLASS_MAP = {
    "boat": "ship",

    "car": "vehicle_convoy",
    "truck": "vehicle_convoy",
    "bus": "vehicle_convoy",
    "motorcycle": "vehicle_convoy",

    "traffic light": "traffic_light",
    "stop sign": "stop_sign",
    "fire hydrant": "fire_hydrant",

    "person": "person",

    "airplane": "unknown_aircraft",
    "train": "railway",

    "ship": "ship",
    "vehicle_convoy": "vehicle_convoy",
}


MISSION_RULES = {

    "MARITIME_SURVEILLANCE": {
        "ship": ("HIGH", 0.90),
    },

    "URBAN_MONITORING": {
        "vehicle_convoy": ("HIGH", 0.90),
        "person": ("MEDIUM", 0.65),
        "traffic_light": ("MEDIUM", 0.60),
        "bicycle": ("LOW", 0.30),
        "stop_sign": ("LOW", 0.30),
        "fire_hydrant": ("LOW", 0.25),
    },

    "DISASTER_RESPONSE": {
        "person": ("CRITICAL", 1.00),
        "vehicle_convoy": ("HIGH", 0.85),
        "wildfire": ("CRITICAL", 1.00),
        "flood": ("CRITICAL", 1.00),
        "landslide": ("CRITICAL", 1.00),
        "ship": ("MEDIUM", 0.60),
        "unknown_aircraft": ("HIGH", 0.85),
        "railway": ("MEDIUM", 0.60),
    },

    "WILDFIRE_RESPONSE": {
        "person": ("HIGH", 0.85),
        "vehicle_convoy": ("MEDIUM", 0.60),
        "unknown_aircraft": ("MEDIUM", 0.60),
    },

    "GENERAL": {
        "ship": ("MEDIUM", 0.60),
        "vehicle_convoy": ("MEDIUM", 0.60),
        "person": ("MEDIUM", 0.60),
        "traffic_light": ("LOW", 0.30),
        "unknown_aircraft": ("MEDIUM", 0.60),
        "railway": ("MEDIUM", 0.60),
    },
}


def normalize_class(name):
    name = str(name).strip().lower()
    return CLASS_MAP.get(name, name)


def package_for_confidence(confidence):
    """
    Evidence packet selected from the mission confidence.

    Returns:
        package_name, packet_size_kb
    """

    percent = confidence * 100.0

    if percent >= 90:
        return "METADATA_ONLY", 1

    if percent >= 60:
        return "METADATA_PLUS_THUMBNAIL", 4

    if percent >= 30:
        return "METADATA_PLUS_IMAGE_CROP", 8

    return "METADATA_PLUS_FULL_IMAGE", 20


def run_mission_scheduler(
    detections,
    bandwidth_kb=20,
    mission_mode="MARITIME_SURVEILLANCE",
):

    if mission_mode not in MISSION_RULES:
        mission_mode = "GENERAL"

    candidates = []

    # ========================================================
    # MISSION FILTER + PRIORITY
    # ========================================================

    for detection in detections:

        raw_type = detection.get(
            "target_class",
            detection.get(
                "type",
                "unknown"
            )
        )

        object_type = normalize_class(
            raw_type
        )

        confidence = float(
            detection.get(
                "confidence_score",
                detection.get(
                    "confidence",
                    0.0
                )
            )
        )

        confidence = max(
            0.0,
            min(
                1.0,
                confidence
            )
        )

        mission_objects = MISSION_RULES[
            mission_mode
        ]

        # ----------------------------------------------------
        # HARD MISSION GATE
        # ----------------------------------------------------

        if object_type not in mission_objects:

            candidates.append({

                "target_class": raw_type,

                "type": object_type,

                "confidence": confidence,

                "priority": "DISCARD",

                "score": 0.0,

                "data_package": "NONE",

                "data_cost_kb": 0,

                "decision": "DISCARD",

                "bounding_box":
                    detection.get(
                        "bounding_box",
                        []
                    ),

                "source_image":
                    detection.get(
                        "source_image",
                        ""
                    ),

                "crop_url":
                    detection.get(
                        "crop_url",
                        ""
                    ),

                "reason":
                    (
                        f"{object_type} is not relevant "
                        f"to the {mission_mode.replace('_', ' ').title()} "
                        f"mission."
                    ),
            })

            continue

        # ----------------------------------------------------
        # Mission priority
        # ----------------------------------------------------

        priority, priority_weight = (
            mission_objects[
                object_type
            ]
        )

        # ----------------------------------------------------
        # Mission score
        #
        # Importance + confidence.
        # ----------------------------------------------------

        score = (
            0.60 * priority_weight
            +
            0.40 * confidence
        )

        score = max(
            0.0,
            min(
                1.0,
                score
            )
        )

        package, cost_kb = (
            package_for_confidence(
                confidence
            )
        )

        candidates.append({

            "target_class":
                raw_type,

            "type":
                object_type,

            "confidence":
                round(
                    confidence,
                    4
                ),

            "priority":
                priority,

            "score":
                round(
                    score,
                    4
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
                    []
                ),

            "source_image":
                detection.get(
                    "source_image",
                    ""
                ),

            "crop_url":
                detection.get(
                    "crop_url",
                    ""
                ),

            "reason":
                (
                    f"{object_type} is relevant to the "
                    f"{mission_mode.replace('_', ' ').title()} "
                    f"mission."
                ),
        })

    # ========================================================
    # SORT BY VALUE
    # ========================================================

    candidates.sort(
        key=lambda item:
            item["score"],
        reverse=True
    )

    # ========================================================
    # SHARED BANDWIDTH
    # ========================================================

    remaining = float(
        bandwidth_kb
    )

    sent = []
    discarded = []

    for item in candidates:

        if item["decision"] == "DISCARD":

            discarded.append(
                item
            )

            continue

        if (
            item["data_cost_kb"]
            <=
            remaining
        ):

            item["decision"] = (
                "TRANSMIT"
            )

            item["reason"] = (
                f"{item['priority']} mission priority; "
                f"packet fits within remaining "
                f"downlink budget."
            )

            remaining -= (
                item["data_cost_kb"]
            )

            sent.append(
                item
            )

        else:

            item["decision"] = (
                "DISCARD"
            )

            item["reason"] = (
                "Mission-relevant target, but the "
                "remaining downlink budget cannot "
                "accommodate its evidence packet."
            )

            discarded.append(
                item
            )

    # ========================================================
    # RANKING
    # ========================================================

    sent.sort(
        key=lambda x:
            x["score"],
        reverse=True
    )

    discarded.sort(
        key=lambda x:
            x["score"],
        reverse=True
    )

    for rank, item in enumerate(
        sent,
        start=1
    ):
        item["rank"] = rank

    for rank, item in enumerate(
        discarded,
        start=1
    ):
        item["rank"] = rank

    return {

        "sent":
            sent,

        "discarded":
            discarded,

        "remaining_bandwidth_kb":
            round(
                remaining,
                2
            ),

        "mission_mode":
            mission_mode,
    }


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    detections = [

        {
            "target_class":
                "boat",

            "confidence_score":
                0.90,

            "bounding_box":
                [1, 2, 3, 4],

            "source_image":
                "marine.jpg",
        },

        {
            "target_class":
                "car",

            "confidence_score":
                0.90,

            "bounding_box":
                [5, 6, 7, 8],

            "source_image":
                "urban.jpg",
        },

        {
            "target_class":
                "traffic light",

            "confidence_score":
                0.70,

            "bounding_box":
                [9, 10, 11, 12],

            "source_image":
                "urban.jpg",
        },
    ]

    print("\n=== MARITIME ===")

    result = run_mission_scheduler(
        detections,
        bandwidth_kb=20,
        mission_mode="MARITIME_SURVEILLANCE",
    )

    for item in (
        result["sent"]
        +
        result["discarded"]
    ):

        print(
            item["target_class"],
            "=>",
            item["decision"],
            "|",
            item["reason"]
        )

    print("\n=== URBAN ===")

    result = run_mission_scheduler(
        detections,
        bandwidth_kb=20,
        mission_mode="URBAN_MONITORING",
    )

    for item in (
        result["sent"]
        +
        result["discarded"]
    ):

        print(
            item["target_class"],
            "=>",
            item["decision"],
            "|",
            item["reason"]
        )