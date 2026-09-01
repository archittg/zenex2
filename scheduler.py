# scheduler.py
# ----------------------------------------------------
# If bandwidth is limited, picks the BEST items first
# (highest score) and drops the rest once budget runs out.
# ----------------------------------------------------

import json
from decision import make_decision


def schedule_transmissions(detections: list, bandwidth_kb: int, mission_mode: str = None) -> dict:
    """
    detections: list of {"type": ..., "confidence": ...}
    bandwidth_kb: total KB available to send this pass

    Returns: {"sent": [...], "discarded": [...]}
    """
    # Step 1: score every detection using decision.py
    scored = [make_decision(d, mission_mode=mission_mode) for d in detections]

    # Step 2: sort by score, highest first (best stuff goes first)
    scored.sort(key=lambda d: d["score"], reverse=True)

    sent = []
    discarded = []
    remaining_budget = bandwidth_kb

    # Step 3: greedily add to "sent" until budget runs out
    for item in scored:
        # Never send items that were already marked DISCARD by decision.py
        if item["decision"] == "DISCARD":
            discarded.append(item)
            continue

        if item["data_cost_kb"] <= remaining_budget:
            sent.append(item)
            remaining_budget -= item["data_cost_kb"]
        else:
            discarded.append(item)

    return {"sent": sent, "discarded": discarded}


if __name__ == "__main__":
    # Same test set as decision.py — types exist in MARITIME_SURVEILLANCE table
    my_detections = [
        {"type": "unknown_ship", "confidence": 0.97},
        {"type": "oil_spill", "confidence": 0.85},
        {"type": "fishing_boat", "confidence": 0.92},
        {"type": "ship", "confidence": 0.70},
        {"type": "small_boat", "confidence": 0.50},
        {"type": "cloud", "confidence": 0.90},
    ]

    # Demo mission mode + a tight bandwidth budget so you can see
    # the scheduler actually drop something due to running out of KB
    MISSION_MODE = "MARITIME_SURVEILLANCE"
    BANDWIDTH_KB = 8

    result = schedule_transmissions(my_detections, bandwidth_kb=BANDWIDTH_KB, mission_mode=MISSION_MODE)

    print("SENT:")
    for item in result["sent"]:
        print(" ", item)

    print("\nDISCARDED (bandwidth ran out or low priority):")
    for item in result["discarded"]:
        print(" ", item)

    # Save the final deliverable for Person 3 (Simulator)
    with open("final_output.json", "w") as f:
        json.dump(result["sent"], f, indent=2)

    print("\nSaved final_output.json with", len(result["sent"]), "items.")