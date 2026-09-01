import os
from ultralytics import YOLO

# Automatically detect any image file in your CODING folder
image_files = [f for f in os.listdir('.') if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

if not image_files:
    print("[-] Error: No image files found in this folder. Make sure your picture is here!")
else:
    target_image = image_files[0]
    print(f"[*] Found image file: {target_image}")
    
    # Load the local YOLO model and run inference
    model = YOLO("yolov8n.pt")
    results = model(target_image)
    
    print("Inference completed successfully!")
    for r in results:
        print(f"Detected {len(r.boxes)} objects in {target_image}.")