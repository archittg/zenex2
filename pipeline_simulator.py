import time
import os
from ultralytics import YOLO
import json

def run_pipeline_simulation():
    print("==================================================")
    print(" SATELLITE EDGE-AI DOWNLINK SIMULATOR INITIALIZED ")
    print("==================================================")
    
    start_time = time.time()
    
    # Step 1: Run On-board AI Inference locally
    model = YOLO("yolov8n.pt")
    image_target = "test.jpg.jpeg"
    
    if not os.path.exists(image_target):
        image_target = "test.jpg"  # Fallback check
        
    results = model.predict(source=image_target, imgsz=640, conf=0.25, verbose=False)
    r = results[0]
    
    # Step 2: Extract structured text metadata payload
    metadata_packet = []
    for i in range(len(r.boxes)):
        class_id = int(r.boxes.cls[i].item())
        class_name = r.names[class_id]
        confidence = float(r.boxes.conf[i].item())
        coords = r.boxes.xyxy[i].tolist()
        
        metadata_packet.append({
            "target_class": class_name,
            "confidence_score": round(confidence, 4),
            "bounding_box": [round(c, 2) for c in coords]
        })

    output_json = "telemetry_metadata.json"
    with open(output_json, "w") as f:
        json.dump(metadata_packet, f, indent=2)
        
    end_time = time.time()
    processing_latency = end_time - start_time
    
    # Step 3: Compute bandwidth optimization metrics
    raw_size_bytes = os.path.getsize(image_target) if os.path.exists(image_target) else 170000
    metadata_size_bytes = os.path.getsize(output_json)
    bandwidth_saved_percent = ((raw_size_bytes - metadata_size_bytes) / raw_size_bytes) * 100

    print("\n--- DOWNLINK TELEMETRY REPORT ---")
    print(f"Processing Latency   : {processing_latency:.4f} seconds")
    print(f"Raw Image Footprint  : {raw_size_bytes} bytes")
    print(f"Metadata Payload Size: {metadata_size_bytes} bytes")
    print(f"Bandwidth Efficiency : {bandwidth_saved_percent:.2f}% reduction achieved")
    print("--------------------------------------------------")

if __name__ == "__main__":
    run_pipeline_simulation()