from pathlib import Path
from datetime import datetime, timezone
import uuid

import cv2

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


# ============================================================
# DIRECTORIES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "yolov8n.pt"

RUNTIME_DIR = BASE_DIR / "runtime"
ANNOTATED_DIR = RUNTIME_DIR / "annotated"
CROPS_DIR = RUNTIME_DIR / "crops"

ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
CROPS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# YOLO MODEL
# ============================================================

_model = None


def get_model():
    """
    Load YOLOv8n once and reuse it.
    """

    global _model

    if YOLO is None:
        raise RuntimeError(
            "Ultralytics is not installed. "
            "Run: python -m pip install -r requirements.txt"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"YOLO model not found at:\n{MODEL_PATH}"
        )

    if _model is None:
        print(f"Loading YOLO model: {MODEL_PATH}")

        _model = YOLO(str(MODEL_PATH))

        print("YOLOv8n loaded successfully.")

    return _model


# ============================================================
# CONFIDENCE POLICY
# ============================================================

def transform_confidence(raw_confidence: float) -> float:
    """
    Convert YOLO confidence into the confidence score used
    by our onboard mission system.

    Input:
        raw_confidence = 0.0 - 1.0

    Output:
        confidence = 0.0 - 1.0

    Policy:

        x < 30
            3x

        30 <= x < 40
            (50 + 3x) / 2

        40 <= x < 50
            2x

        x >= 50
            (100 + x) / 2
    """

    x = raw_confidence * 100.0

    if x < 30:
        confidence_percent = 3 * x

    elif x < 40:
        confidence_percent = (50 + 3 * x) / 2

    elif x < 50:
        confidence_percent = 2 * x

    else:
        confidence_percent = (100 + x) / 2

    # Safety limit.
    confidence_percent = max(
        0.0,
        min(100.0, confidence_percent)
    )

    return confidence_percent / 100.0


# ============================================================
# IMAGE ANALYSIS
# ============================================================

def analyze_image(
    image_path: str,
    confidence_threshold: float = 0.25
) -> dict:

    image_path = Path(image_path)

    # --------------------------------------------------------
    # Validate image
    # --------------------------------------------------------

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    if image_path.suffix.lower() not in {
        ".jpg",
        ".jpeg",
        ".png"
    }:
        raise ValueError(
            f"Unsupported image format: "
            f"{image_path.suffix}"
        )

    if not 0.01 <= confidence_threshold <= 0.99:
        raise ValueError(
            "Confidence threshold must be between "
            "0.01 and 0.99."
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = get_model()

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    original_image = cv2.imread(
        str(image_path)
    )

    if original_image is None:
        raise RuntimeError(
            f"Could not read image:\n{image_path}"
        )

    height, width = original_image.shape[:2]

    raw_size_bytes = image_path.stat().st_size

    # --------------------------------------------------------
    # YOLO inference
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print(f"Processing: {image_path.name}")
    print(f"Image: {width} x {height}")
    print(
        f"YOLO threshold: "
        f"{confidence_threshold:.2f}"
    )
    print("=" * 60)

    results = model.predict(
        source=str(image_path),
        imgsz=640,
        conf=confidence_threshold,
        verbose=False
    )

    if not results:
        raise RuntimeError(
            "YOLO returned no result."
        )

    result = results[0]

    # --------------------------------------------------------
    # Annotated image
    # --------------------------------------------------------

    unique_id = uuid.uuid4().hex[:8]

    annotated_filename = (
        f"{image_path.stem}_"
        f"{unique_id}_detected.jpg"
    )

    annotated_path = (
        ANNOTATED_DIR /
        annotated_filename
    )

    annotated_image = result.plot()

    if not cv2.imwrite(
        str(annotated_path),
        annotated_image
    ):
        raise RuntimeError(
            f"Could not save annotated image:\n"
            f"{annotated_path}"
        )

    annotated_url = (
        f"/runtime/annotated/"
        f"{annotated_filename}"
    )

    # --------------------------------------------------------
    # Detections
    # --------------------------------------------------------

    detections = []

    number_of_boxes = len(result.boxes)

    print(
        f"Detections found: "
        f"{number_of_boxes}"
    )

    for i in range(number_of_boxes):

        # ----------------------------------------------------
        # Class
        # ----------------------------------------------------

        class_id = int(
            result.boxes.cls[i].item()
        )

        object_name = str(
            result.names[class_id]
        )

        # ----------------------------------------------------
        # YOLO confidence
        # ----------------------------------------------------

        raw_confidence = float(
            result.boxes.conf[i].item()
        )

        # ----------------------------------------------------
        # Our mission confidence
        # ----------------------------------------------------

        confidence = transform_confidence(
            raw_confidence
        )

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        bbox = [
            int(round(value))
            for value in
            result.boxes.xyxy[i].tolist()
        ]

        x1, y1, x2, y2 = bbox

        # Keep coordinates inside image.
        x1 = max(
            0,
            min(x1, width - 1)
        )

        y1 = max(
            0,
            min(y1, height - 1)
        )

        x2 = max(
            0,
            min(x2, width)
        )

        y2 = max(
            0,
            min(y2, height)
        )

        bbox = [
            x1,
            y1,
            x2,
            y2
        ]

        # ----------------------------------------------------
        # Object crop
        # ----------------------------------------------------

        crop_url = ""

        if x2 > x1 and y2 > y1:

            crop = original_image[
                y1:y2,
                x1:x2
            ]

            crop_filename = (
                f"{image_path.stem}_"
                f"{unique_id}_"
                f"crop_{i + 1}.jpg"
            )

            crop_path = (
                CROPS_DIR /
                crop_filename
            )

            if cv2.imwrite(
                str(crop_path),
                crop
            ):

                crop_url = (
                    f"/runtime/crops/"
                    f"{crop_filename}"
                )

        # ----------------------------------------------------
        # Detection object
        #
        # IMPORTANT:
        # Only our final confidence is exposed.
        # ----------------------------------------------------

        detection = {

            "type":
                object_name,

            "confidence":
                round(
                    confidence,
                    4
                ),

            "bbox":
                bbox,

            "source_image":
                image_path.name,

            "crop_url":
                crop_url,
        }

        detections.append(
            detection
        )

        # ----------------------------------------------------
        # Terminal information
        # ----------------------------------------------------

        print(
            f"[{i + 1}] "
            f"{object_name:<15} "
            f"Confidence="
            f"{confidence * 100:.2f}% "
            f"BBox={bbox}"
        )

    # --------------------------------------------------------
    # Final AI result
    # --------------------------------------------------------

    return {

        "image_id":
            image_path.name,

        "timestamp":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "raw_size_bytes":
            raw_size_bytes,

        "image_width":
            width,

        "image_height":
            height,

        "annotated_image_url":
            annotated_url,

        "detections":
            detections,

        "cloud_cover":
            0.20,
    }


# ============================================================
# COMPATIBILITY WRAPPER
# ============================================================

def run_pipeline(
    image_path: str,
    confidence_threshold: float = 0.25
):
    return analyze_image(
        image_path,
        confidence_threshold
    )


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:

        print(
            "\nUsage:\n"
            "python detector.py image.jpg\n"
        )

        sys.exit(1)

    image = sys.argv[1]

    threshold = (
        float(sys.argv[2])
        if len(sys.argv) > 2
        else 0.25
    )

    try:

        result = analyze_image(
            image,
            threshold
        )

        print()
        print("=" * 60)
        print("FINAL DETECTION RESULTS")
        print("=" * 60)

        for i, detection in enumerate(
            result["detections"],
            start=1
        ):

            print(
                f"{i}. "
                f"{detection['type']} | "
                f"Confidence: "
                f"{detection['confidence'] * 100:.2f}% | "
                f"BBox: "
                f"{detection['bbox']}"
            )

        print("=" * 60)

    except Exception as exc:

        print()
        print("DETECTION FAILED")
        print(exc)

        sys.exit(1)