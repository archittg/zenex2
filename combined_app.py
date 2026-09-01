import streamlit as st
import os
import json
from PIL import Image
from ultralytics import YOLO

from pipeline import adapt_detection
from scheduler import schedule_transmissions
from mission_modes import MISSION_MODES

st.set_page_config(
    page_title="Satellite Edge-AI Downlink Dashboard",
    page_icon="🛰️",
    layout="wide"
)


@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")


model = load_model()

st.sidebar.header("Edge-AI Configuration")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", min_value=0.01, max_value=1.00, value=0.25, step=0.01,
    help="Adjust model sensitivity to filter out background objects."
)
boost_low_confidence = st.sidebar.checkbox(
    "Boost detection confidence (demo mode)",
    value=True,
    help="For demo purposes only — rescales every detection's confidence using a tiered "
         "formula: <30% is tripled, 30-40% uses (x*3+50)/2, 40-50% is doubled, and 50%+ "
         "is averaged with 100. This does NOT make the AI model more accurate."
)


def boost_confidence(confidence: float, enabled: bool) -> float:
    """Demo-mode only: rescales raw model confidence using a tiered formula.
    Does not reflect real model accuracy — see sidebar tooltip.

    Tiers (working in percentage terms, then converted back to 0-1):
      < 30%      -> confidence * 3
      30% - 40%  -> (confidence * 3 + 50) / 2
      40% - 50%  -> confidence * 2
      >= 50%     -> (confidence + 100) / 2
    Result is capped at 100%.
    """
    if not enabled:
        return confidence

    pct = confidence * 100
    if pct < 30:
        boosted_pct = pct * 3
    elif pct < 40:
        boosted_pct = (pct * 3 + 50) / 2
    elif pct < 50:
        boosted_pct = pct * 2
    else:
        boosted_pct = (pct + 100) / 2

    boosted_pct = min(boosted_pct, 100)
    return round(boosted_pct / 100, 4)


st.sidebar.header("Downlink Configuration")
mission_mode = st.sidebar.selectbox(
    "Mission Mode",
    options=list(MISSION_MODES.keys()),
    index=list(MISSION_MODES.keys()).index("MARITIME_SURVEILLANCE"),
    help="Same object gets a different priority depending on the active mission."
)
bandwidth_kb = st.sidebar.slider(
    "Bandwidth Budget (KB per pass)", min_value=1, max_value=100, value=20, step=1,
    help="Total downlink budget for this satellite pass. Lower-scoring detections get dropped once this runs out."
)

st.title("🛰️ Satellite Edge-AI & Bandwidth Optimizer")
st.markdown(
    "Upload one or more images to simulate a satellite pass: on-orbit AI detection across "
    "every frame, mission-based priority scoring, and ONE shared bandwidth budget for the whole pass."
)

uploaded_files = st.file_uploader(
    "Upload Satellite Imagery (JPG, PNG) — you can select multiple files at once",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if uploaded_files:
    all_raw_detections = []   # pooled across every image, for the scheduler
    total_raw_size = 0

    # ---------------- Step 1: Raw image + AI detection (Person 1), per image ----------------
    st.markdown("---")
    st.subheader(f"Onboard Edge-AI Detections — {len(uploaded_files)} image(s) this pass")

    for idx, uploaded_file in enumerate(uploaded_files):
        img_path = f"temp_uploaded_image_{idx}.jpg"
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        raw_size = os.path.getsize(img_path)
        total_raw_size += raw_size

        with st.expander(f"📷 {uploaded_file.name} ({raw_size/1024:.2f} KB)", expanded=(idx == 0)):
            col1, col2 = st.columns(2)

            with col1:
                image = Image.open(img_path)
                st.image(image, caption="Captured Frame", use_container_width=True)

            with col2:
                results = model.predict(source=img_path, imgsz=640, conf=confidence_threshold, verbose=False)
                r = results[0]

                res_plotted = r.plot()
                st.image(res_plotted, caption="Processed Telemetry Visual", channels="BGR", use_container_width=True)

                # No class whitelist here — every detection gets passed to the
                # priority engine below, which decides relevance per mission mode.
                image_detections = []
                for i in range(len(r.boxes)):
                    class_id = int(r.boxes.cls[i].item())
                    class_name = r.names[class_id]
                    raw_confidence = float(r.boxes.conf[i].item())
                    confidence = boost_confidence(raw_confidence, boost_low_confidence)
                    coords = r.boxes.xyxy[i].tolist()
                    image_detections.append({
                        "target_class": class_name,
                        "confidence_score": round(confidence, 4),
                        "bounding_box": [round(c, 2) for c in coords],
                        "source_image": uploaded_file.name,
                    })

                if image_detections:
                    st.json(image_detections)
                else:
                    st.info("No objects detected at this threshold.")
                all_raw_detections.extend(image_detections)

        os.remove(img_path)

    # ---------------- Step 2: Decision Logic (Person 2) — one shared budget across all images ----------------
    st.markdown("---")
    st.subheader(f"📋 Priority & Downlink Decision — Mode: {mission_mode}")
    st.caption(
        f"{len(all_raw_detections)} total detection(s) pooled from {len(uploaded_files)} image(s), "
        f"competing for a single {bandwidth_kb} KB downlink budget this pass."
    )

    if not all_raw_detections:
        st.info("No objects detected across any uploaded image at the current confidence threshold.")
    else:
        adapted = [adapt_detection(d) for d in all_raw_detections]
        result = schedule_transmissions(adapted, bandwidth_kb=bandwidth_kb, mission_mode=mission_mode)

        # Add explicit rank numbers so it's visually clear both lists are
        # ordered best-score-first (SENT continues where it would rank,
        # DISCARDED picks up right after)
        sent_ranked = [{"rank": i + 1, **item} for i, item in enumerate(result["sent"])]
        discarded_ranked = [{"rank": len(result["sent"]) + i + 1, **item} for i, item in enumerate(result["discarded"])]

        sent_col, discard_col = st.columns(2)

        with sent_col:
            st.markdown("**✅ SENT (within bandwidth budget)**")
            if sent_ranked:
                st.table(sent_ranked)
            else:
                st.write("Nothing qualified for transmission this pass.")

        with discard_col:
            st.markdown("**🗑️ DISCARDED (low priority or budget ran out)**")
            if discarded_ranked:
                st.table(discarded_ranked)
            else:
                st.write("Nothing discarded — everything fit.")

        # Save the deliverable for Person 3, same as pipeline.py does on the command line
        with open("final_output.json", "w") as f:
            json.dump(result["sent"], f, indent=2)
        st.success(f"Saved final_output.json with {len(result['sent'])} item(s) for Person 3 (Simulator).")

    # ---------------- Step 3: Bandwidth Efficiency Report ----------------
    st.markdown("---")
    st.subheader("📡 Downlink Bandwidth Efficiency Report")

    metadata_size = len(json.dumps(all_raw_detections).encode("utf-8"))
    saving_percent = ((total_raw_size - metadata_size) / total_raw_size) * 100 if total_raw_size else 0

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Raw Image Transmit Cost (all images)", f"{total_raw_size/1024:.2f} KB")
    metric_col2.metric("Metadata Transmit Cost", f"{metadata_size} bytes")
    metric_col3.metric("Bandwidth Saved (metadata vs raw)", f"{saving_percent:.2f}%", delta="Optimized")
