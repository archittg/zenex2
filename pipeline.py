# pipeline.py
# ----------------------------------------------------
# Bridges Person 1 (AI/Vision) -> Person 2 (Decision Logic) -> Person 3 (Simulator)
#
# Person 1's YOLO detector (metadata_formatter.py / pipeline_simulator.py)
# writes files shaped like:
#   {"target_class": "boat", "confidence_score": 0.85, "bounding_box": [x1,y1,x2,y2]}
#
# Person 2's decision.py / scheduler.py expect:
#   {"type": "ship", "confidence": 0.85}
#
# adapt_detection() does TWO jobs, not just a field rename:
#   1. Renames fields (target_class -> type, confidence_score -> confidence)
#   2. Translates Person 1's generic YOLO/COCO class names into THIS
#      project's satellite object-type vocabulary (priority.py / mission_modes.py).
#      Without this step, "boat" would never match "ship" in your tables
#      and would silently fall back to LOW priority every time.
# ----------------------------------------------------

import json
import os
from scheduler import schedule_transmissions

# Person 1's model is a stock YOLOv8n, trained on the generic COCO dataset —
# it has no idea this is a satellite project. It only knows COCO's 80 labels
# (car, boat, airplane, laptop, etc). This table translates the labels that
# are actually relevant to satellite surveillance into our vocabulary.
# Anything not listed here (laptop, chair, dog, ...) is passed through as-is,
# which means priority.py/mission_modes.py will fall back to "LOW" for it —
# that's correct: those objects are irrelevant to this mission.
CLASS_TYPE_MAP = {
    "boat": "ship",
    "car": "vehicle_convoy",
    "truck": "vehicle_convoy",
    "bus": "vehicle_convoy",
    "motorcycle": "vehicle_convoy",
    "airplane": "unknown_aircraft",
    "train": "railway",
}


def adapt_detection(raw: dict) -> dict:
    """
    Converts ONE of Person 1's raw detections into the
    {"type": ..., "confidence": ...} format decision.py expects.
    Tolerant of minor key-naming differences in case Person 1's
    output format shifts slightly later.
    """
    class_name = raw.get("target_class", raw.get("class", raw.get("type", "unknown_thing")))
    confidence = raw.get("confidence_score", raw.get("confidence", raw.get("score", 0.0)))

    mapped_type = CLASS_TYPE_MAP.get(class_name, class_name)

    return {
        "type": mapped_type,
        "confidence": round(float(confidence), 4),
    }


def load_detections(path: str) -> list:
    """Reads Person 1's raw output file and adapts every detection in it."""
    with open(path, "r") as f:
        raw_detections = json.load(f)
    return [adapt_detection(d) for d in raw_detections]


def run_pipeline(input_path: str, bandwidth_kb: int = 20, mission_mode: str = "MARITIME_SURVEILLANCE") -> dict:
    detections = load_detections(input_path)

    print(f"[+] Loaded {len(detections)} detection(s) from {input_path}")
    for original, adapted in zip(json.load(open(input_path)), detections):
        orig_class = original.get("target_class", "?")
        print(f"    {orig_class!r} -> type={adapted['type']!r}, confidence={adapted['confidence']}")

    result = schedule_transmissions(detections, bandwidth_kb=bandwidth_kb, mission_mode=mission_mode)

    print("\nSENT (goes to Person 3):")
    for item in result["sent"]:
        print(" ", item)

    print("\nDISCARDED:")
    for item in result["discarded"]:
        print(" ", item)

    with open("final_output.json", "w") as f:
        json.dump(result["sent"], f, indent=2)

    print(f"\n[+] Saved final_output.json with {len(result['sent'])} item(s).")
    return result


if __name__ == "__main__":
    # Look for whichever file Person 1's scripts actually produced.
    candidates = ["telemetry_metadata.json", "satellite_metadata.json"]
    input_file = next((c for c in candidates if os.path.exists(c)), None)

    if input_file is None:
        print("[-] No input file found. Ask Person 1 to run metadata_formatter.py")
        print("    or pipeline_simulator.py first to generate one of:")
        for c in candidates:
            print("     -", c)
    else:
        run_pipeline(input_file)
