import random
from schemas import MissionSummary

class SatelliteSimulator:
    def __init__(self):
        self.reset()

    def reset(self):
        self.battery = 78.0
        self.bandwidth_kbps = 500.0
        self.cpu_load = 10.0
        self.ground_link = True
        self.total_raw_data = 0
        self.total_transmitted_data = 0
        self.energy_used = 0.0
        self.transmission_time_ms = 0.0

    def apply_pass(self, raw_bytes: int, optimized_bytes: int, inference_ms: float):
        self.cpu_load = min(98.0, max(8.0, 15.0 + inference_ms / 20.0 + random.uniform(0, 15)))
        self.ground_link = random.random() > 0.03
        self.total_raw_data += int(max(0, raw_bytes))
        self.total_transmitted_data += int(max(0, optimized_bytes))

        # Same simplified model as the supplied simulator:
        # 0.1 J baseline + 0.05 J per transmitted KB.
        self.energy_used = 0.1 + (optimized_bytes / 1024.0) * 0.05
        self.transmission_time_ms = (
            (optimized_bytes * 8.0) / self.bandwidth_kbps
            if self.bandwidth_kbps else 0.0
        )
        self.battery = max(0.0, self.battery - self.energy_used)
        return self.status()

    def trigger_pass(self):
        self.cpu_load = random.uniform(45.0, 85.0)
        self.battery = max(0.0, self.battery - 0.05)
        self.ground_link = True
        return self.status()

    def status(self):
        return {
            "battery_percent": round(self.battery, 2),
            "bandwidth_available_kbps": round(self.bandwidth_kbps, 2),
            "cpu_load_percent": round(self.cpu_load, 2),
            "ground_link_active": self.ground_link,
            "raw_data_size_bytes": self.total_raw_data,
            "metadata_size_bytes": self.total_transmitted_data,
            "energy_used": round(self.energy_used, 4),
            "transmission_time_ms": round(self.transmission_time_ms, 3),
        }

satellite = SatelliteSimulator()
