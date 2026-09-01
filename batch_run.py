import os
import requests

BASE_URL = "http://127.0.0.1:8000"

def run_orbit_folder(folder: str, mission_mode="MARITIME_SURVEILLANCE", budget_kb=20, confidence=0.25):
    images = [
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if not images:
        print("No images found.")
        return

    files = []
    handles = []
    try:
        for path in images:
            h = open(path, "rb")
            handles.append(h)
            files.append(("files", (os.path.basename(path), h, "image/jpeg")))
        response = requests.post(
            f"{BASE_URL}/api/analyze",
            files=files,
            data={
                "mission_mode": mission_mode,
                "bandwidth_kb": budget_kb,
                "confidence_threshold": confidence,
            },
            timeout=180,
        )
        response.raise_for_status()
        data = response.json()
        print("Orbit pass complete.")
        print("Detections:", data["summary"]["total_detections"])
        print("Transmitted:", data["summary"]["transmitted"])
        print("Discarded:", data["summary"]["discarded"])
        print("Bandwidth reduction:", data["summary"]["bandwidth_reduction_percent"], "%")
    finally:
        for h in handles:
            h.close()

if __name__ == "__main__":
    run_orbit_folder(os.path.join(os.path.dirname(__file__), "orbit_samples"))
