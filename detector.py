# ai_engine/detector.py
import os
import time
import requests
from datetime import datetime
from PIL import Image
from ultralytics import YOLO

# Load lightweight pre-trained YOLO model (automatically downloads on first run)
model = YOLO("yolov8n.pt")

BACKEND_URL = "http://127.0.0.1:8000"

def analyze_image(image_path: str):
    """Processes an image, extracts detected targets, and simulates cloud cover analysis."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    # 1. Get image metadata
    raw_size_bytes = os.path.getsize(image_path)
    image_id = os.path.basename(image_path)
    timestamp = datetime.utcnow().isoformat()

    # 2. Run YOLO Inference
    results = model(image_path)
    
    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])
            bbox = [int(x) for x in box.xyxy[0].tolist()]  # [xmin, ymin, xmax, ymax]
            
            detections.append({
                "type": label,
                "confidence": round(conf, 2),
                "bbox": bbox
            })

    # 3. Simulate cloud cover estimation (Percentage)
    # In real pipeline: calculated via brightness/white pixel ratio thresholding
    cloud_cover = round(min(1.0, max(0.0, (len(detections) * 0.1) % 0.8)), 2)

    # 4. Form JSON matching our AIResult schema
    ai_result = {
        "image_id": image_id,
        "timestamp": timestamp,
        "detections": detections,
        "cloud_cover": cloud_cover,
        "raw_size_bytes": raw_size_bytes
    }

    return ai_result

def run_pipeline(image_path: str):
    print(f"🛰️ Processing satellite frame: {image_path}...")
    ai_result = analyze_image(image_path)
    
    print("✅ AI Extraction Complete:")
    print(f"   - Target Detections: {len(ai_result['detections'])}")
    print(f"   - Estimated Cloud Cover: {ai_result['cloud_cover'] * 100}%")
    print(f"   - Raw Size: {ai_result['raw_size_bytes'] / 1024:.2f} KB")

    # Basic Mission Decision (Person 2 rule logic shortcut for testing)
    highest_priority_detection = ai_result['detections'][0]['type'] if ai_result['detections'] else "None"
    highest_conf = ai_result['detections'][0]['confidence'] if ai_result['detections'] else 0.0

    # Rules: High cloud cover -> DISCARD; Detections found -> TRANSMIT
    if ai_result['cloud_cover'] > 0.70:
        decision = "DISCARD"
        priority = 0.1
    elif len(ai_result['detections']) > 0:
        decision = "TRANSMIT"
        priority = 0.9
    else:
        decision = "STORE"
        priority = 0.4

    mission_decision = {
        "detection_type": highest_priority_detection,
        "confidence": highest_conf,
        "priority": priority,
        "decision": decision
    }

    # 5. Send data payload to backend server
    payload = {
        "ai_result": ai_result,
        "decision": mission_decision
    }

    try:
        response = requests.post(f"{BACKEND_URL}/process-image", json=payload)
        print(f"📡 Ground Transmission Response: {response.json()}")
    except Exception as e:
        print(f"❌ Failed to connect to Backend API: {e}")

if __name__ == "__main__":
    # Test with a dummy image path or sample image
    import sys
    test_image = sys.argv[1] if len(sys.argv) > 1 else "sample.jpg"
    run_pipeline(test_image)