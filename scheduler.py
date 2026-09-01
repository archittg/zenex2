# scheduler.py
# ----------------------------------------------------
# If bandwidth is limited, picks the BEST items first
# (highest score) and drops the rest once budget runs out.
# ----------------------------------------------------

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
    fake_detections = [
        {"type": "wildfire", "confidence": 0.99},
        {"type": "ship", "confidence": 0.94},
        {"type": "ship", "confidence": 0.6},
        {"type": "building", "confidence": 0.55},
        {"type": "cloud", "confidence": 0.8},
    ]

    result = schedule_transmissions(fake_detections, bandwidth_kb=10)

    print("SENT:")
    for item in result["sent"]:
        print(" ", item)

    print("\nDISCARDED (bandwidth ran out or low priority):")
    for item in result["discarded"]:
        print(" ", item)