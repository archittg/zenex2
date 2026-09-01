from ultralytics import YOLO

# Load your baseline model
model = YOLO("yolov8n.pt")

# 1. Export to ONNX format
model.export(format="onnx", imgsz=640)

# 2. Export with INT8 Quantization (optimized for edge hardware)
model.export(format="onnx", int8=True, imgsz=640)