# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from schemas import AIResult, MissionDecision, SimulationState
from simulator import satellite

app = FastAPI(title="Space Edge AI Simulator API")

# Allow the React frontend to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/satellite/status", response_model=SimulationState)
def get_status():
    """Returns the current physical state of the simulated satellite."""
    satellite.update_state()
    return SimulationState(
        battery_percent=round(satellite.battery, 2),
        bandwidth_available_kbps=satellite.bandwidth_kbps,
        cpu_load_percent=round(satellite.cpu_load, 2),
        ground_link_active=satellite.ground_link,
        raw_data_size_bytes=satellite.total_raw_data,
        metadata_size_bytes=satellite.total_transmitted_data,
        energy_used=0.0,
        transmission_time_ms=0.0
    )

@app.post("/process-image")
def process_image(ai_result: AIResult, decision: MissionDecision):
    """Processes simulated image detections and updates satellite state."""
    if not satellite.ground_link and decision.decision == "TRANSMIT":
        satellite.downlink_queue.append(decision)
        return {"status": "queued", "message": "No ground link. Data stored."}
        
    if decision.decision == "TRANSMIT":
        metadata_size = 2048 
        energy, tx_time = satellite.process_transmission(ai_result.raw_size_bytes, metadata_size)
        
        return {
            "status": "transmitted",
            "energy_used": round(energy, 4),
            "transmission_time_ms": round(tx_time, 2),
            "reduction_percentage": round((1 - (metadata_size / ai_result.raw_size_bytes)) * 100, 4)
        }
        
    return {"status": "discarded_or_stored"}

@app.post("/simulation/start")
def start_sim():
    satellite.is_running = True
    return {"message": "Simulation started"}

@app.post("/simulation/stop")
def stop_sim():
    satellite.is_running = False
    return {"message": "Simulation paused"}

@app.post("/simulation/toggle-ground-link")
def toggle_link():
    satellite.ground_link = not satellite.ground_link
    return {"ground_link_active": satellite.ground_link}