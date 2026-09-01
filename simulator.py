# backend/simulator.py
import random

class SatelliteSimulator:
    def __init__(self):
        self.battery = 78.0  # Starting battery percentage
        self.bandwidth_kbps = 500.0
        self.cpu_load = 10.0
        self.ground_link = True
        self.downlink_queue = []
        self.total_raw_data = 0
        self.total_transmitted_data = 0
        self.is_running = True

    def update_state(self):
        """Simulates environmental changes over time, like ground station visibility."""
        if not self.is_running:
            return
            
        # Drain battery slowly
        self.battery = max(0.0, self.battery - 0.05)
        
        # Fluctuate CPU load
        self.cpu_load = random.uniform(20.0, 85.0)
        
        # Simulate ground station connection toggles
        if random.random() < 0.05:
            self.ground_link = not self.ground_link

    def process_transmission(self, raw_size_bytes: int, metadata_size_bytes: int):
        """Calculates energy and time cost for transmitting data."""
        self.total_raw_data += raw_size_bytes
        self.total_transmitted_data += metadata_size_bytes
        
        # Simple energy model: computation energy + transmission energy
        energy_used = 0.1 + (metadata_size_bytes / 1024) * 0.05
        self.battery -= energy_used
        
        # Calculate latency
        transmission_time_ms = (metadata_size_bytes * 8) / self.bandwidth_kbps 
        
        return energy_used, transmission_time_ms

# Global instance to hold state in memory
satellite = SatelliteSimulator()