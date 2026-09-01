import streamlit as st
import os
import json
from PIL import Image
from ultralytics import YOLO

st.set_page_config(
    page_title="Satellite Edge-AI Telemetry Dashboard",
    page_icon="🛰️",
    layout="wide"
)

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

st.sidebar.header("Edge-AI Configuration")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.01,
    max_value=1.00,
    value=0.25,
    step=0.01,
    help="Adjust model sensitivity to filter out background objects."
)

st.title("🛰️ Satellite Edge-AI & Bandwidth Optimizer")
st.markdown("Upload any random image capture to simulate on-orbit AI detection and downlinked JSON metadata transmission.")

uploaded_file = st.file_uploader("Upload Satellite Imagery (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img_path = "temp_uploaded_image.jpg"
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Raw Image Payload")
        image = Image.open(img_path)
        st.image(image, caption="Captured Frame", use_container_width=True)
        raw_size = os.path.getsize(img_path)
        st.text(f"Footprint Size: {raw_size:,} bytes (~{raw_size/1024:.2f} KB)")

    with col2:
        st.subheader("Onboard Edge-AI Detections")
        results = model.predict(source=img_path, imgsz=640, conf=confidence_threshold, verbose=False)
        r = results[0]
        
        res_plotted = r.plot()
        st.image(res_plotted, caption="Processed Telemetry Visual", channels="BGR", use_container_width=True)
        
        # Whitelist filter to prevent irrelevant false positives (like benches/chairs)
        allowed_classes = ["car", "truck", "bus", "motorcycle", "airplane", "boat", "train"]
        
        metadata_packet = []
        for i in range(len(r.boxes)):
            class_id = int(r.boxes.cls[i].item())
            class_name = r.names[class_id]
            
            if class_name in allowed_classes:
                confidence = float(r.boxes.conf[i].item())
                coords = r.boxes.xyxy[i].tolist()
                metadata_packet.append({
                    "target_class": class_name,
                    "confidence_score": round(confidence, 4),
                    "bounding_box": [round(c, 2) for c in coords]
                })
            
        json_str = json.dumps(metadata_packet, indent=2)
        metadata_size = len(json_str.encode('utf-8'))
        
        st.text(f"Metadata Payload Size: {metadata_size} bytes")
        st.json(metadata_packet)

    st.markdown("---")
    st.subheader("📡 Downlink Bandwidth Efficiency Report")
    
    saving_percent = ((raw_size - metadata_size) / raw_size) * 100
    
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Raw Image Transmit Cost", f"{raw_size/1024:.2f} KB")
    metric_col2.metric("Metadata Transmit Cost", f"{metadata_size} bytes")
    metric_col3.metric("Bandwidth Saved", f"{saving_percent:.2f}%", delta="Optimized")