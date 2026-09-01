import json
import os
from ultralytics import YOLO

# Load the local model and run inference on your image
model = YOLO("yolov8n.pt")
results = model("test.jpg.jpeg")

metadata_packet = []

# Extract bounding boxes, classes, and confidence scores
for r in results:
    boxes = r.boxes
    for i in range(len(boxes)):
        class_id = int(boxes.cls[i].item())
        class_name = r.names[class_id]
        confidence = float(boxes.conf[i].item())
        coords = boxes.xyxy[i].tolist() # [x1, y1, x2, y2]

        item = {
            "target_class": class_name,
            "confidence_score": round(confidence, 4),
            "bounding_box": [round(c, 2) for c in coords]
        }
        metadata_packet.append(item)

# Save the extracted data into a lightweight JSON file
output_json = "satellite_metadata.json"
with open(output_json, "w") as f:
    json.dump(metadata_packet, f, indent=2)

print(f"[+] Metadata successfully compiled and saved to {output_json} ({os.path.getsize(output_json)} bytes)")