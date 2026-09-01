# decision.py
# ----------------------------------------------------
# THE CORE FILE. Combines priority.py, confidence_policy.py,
# and mission_modes.py into ONE final decision per detection.
#
# For each detection, it:
#  1. Looks up priority (default or mission-specific)
#  2. Looks up data package/cost based on confidence
#  3. Computes a score
#  4. Turns the score into TRANSMIT / TRANSMIT_COMPRESSED / DISCARD
# ----------------------------------------------------

from priority import get_priority, DEFAULT_PRIORITY_TABLE
from confidence_policy import get_data_package
from mission_modes import get_priority_for_mode

# Priority label -> numeric weight, used in the score formula
PRIORITY_WEIGHTS = {
    "CRITICAL": 1.0,
    "HIGH": 0.75,
    "MEDIUM": 0.5,
    "LOW": 0.25,
    "DISCARD": 0.0,
}


def compute_score(confidence: float, priority: str) -> float:
    """
    Score = confidence * priority_weight.
    A wildfire at 0.99 confidence scores near 1.0 (send it).
    A cloud at 0.8 confidence scores 0.0 (never send it).
    """
    weight = PRIORITY_WEIGHTS.get(priority, 0.0)
    return round(confidence * weight, 4)


def score_to_decision(score: float) -> str:
    """
    Turns the numeric score into a final action.
    Tune these thresholds as needed.
    """
    if score >= 0.6:
        return "TRANSMIT"
    elif score >= 0.2:
        return "TRANSMIT_COMPRESSED"
    else:
        return "DISCARD"


def make_decision(detection: dict, mission_mode: str = None) -> dict:
    """
    detection: {"type": "ship", "confidence": 0.94}
    mission_mode: optional, e.g. "MARITIME_SURVEILLANCE"

    Returns the full decision dict.
    """
    object_type = detection["type"]
    confidence = detection["confidence"]

    # Step 1: priority (mission-specific if a mode is given, else default)
    if mission_mode:
        priority = get_priority_for_mode(object_type, mission_mode)
    else:
        priority = get_priority(object_type)

    # Step 2: data package based on confidence
    package_info = get_data_package(confidence)

    # Step 3: score
    score = compute_score(confidence, priority)

    # Step 4: final decision
    decision = score_to_decision(score)

    return {
        "type": object_type,
        "confidence": confidence,
        "priority": priority,
        "score": score,
        "data_package": package_info["package"],
        "data_cost_kb": package_info["cost_kb"],
        "decision": decision,
    }


if __name__ == "__main__":
    # Fake test detections — swap these for your own later
    fake_detections = [
        {"type": "ship", "confidence": 0.94},
        {"type": "wildfire", "confidence": 0.99},
        {"type": "building", "confidence": 0.55},
        {"type": "cloud", "confidence": 0.8},
    ]

    print("--- Default mode ---")
    for d in fake_detections:
        print(make_decision(d))

    print("\n--- MARITIME_SURVEILLANCE mode ---")
    for d in fake_detections:
        print(make_decision(d, mission_mode="MARITIME_SURVEILLANCE"))