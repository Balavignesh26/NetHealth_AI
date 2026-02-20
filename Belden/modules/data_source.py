import random
import time
import datetime
from typing import Dict, List
import math

class SyntheticDataSource:
    def __init__(self):
        self.scenarios = {
            "normal": {
                "crc_errors": (0, 5),
                "snr": (25, 30),
                "latency": (10, 30),
                "packet_loss": (0, 0.5),
                "temperature": (35, 45),
                "cpu_load": (5, 15),
                "retransmits": (0, 2)
            },
            "cable_failure": {
                "crc_errors": (400, 500),
                "snr": (10, 15),
                "latency": (350, 450),
                "packet_loss": (3, 7),
                "temperature": (60, 70),
                "cpu_load": (10, 20),
                "retransmits": (15, 25)
            },
            "emi_interference": {
                "crc_errors": (100, 200),
                "snr": (15, 20),
                "latency": (50, 150),
                "packet_loss": (1, 3),
                "temperature": (35, 45),
                "cpu_load": (5, 15),
                "retransmits": (5, 12)
            }
        }
        self.current_scenario = "normal"

    def set_scenario(self, scenario_name: str):
        if scenario_name in self.scenarios:
            self.current_scenario = scenario_name

    def get_metrics(self, device_id: str) -> Dict[str, float]:
        ranges = self.scenarios[self.current_scenario]
        
        # Add some jitter/noise
        metrics = {
            "crc_errors": max(0, random.uniform(*ranges["crc_errors"])),
            "snr": random.uniform(*ranges["snr"]),
            "latency": random.uniform(*ranges["latency"]),
            "packet_loss": max(0, random.uniform(*ranges["packet_loss"])),
            "temperature": random.uniform(*ranges["temperature"]),
            "cpu_load": random.uniform(*ranges["cpu_load"]),
            "retransmits": max(0, random.uniform(*ranges["retransmits"]))
        }
        
        # Profinet/Modbus specific mocks as needed
        metrics["modbus_register_40001"] = random.uniform(220, 230) # Voltage mock
        metrics["profinet_io_cycle_time"] = random.uniform(1.0, 2.0) # ms
        
        return metrics

    def generate_time_series(self, device_id: str, minutes: int = 60, interval_sec: int = 60) -> List[Dict]:
        """Generates historical data for training/analysis."""
        history = []
        base_time = datetime.datetime.utcnow() - datetime.timedelta(minutes=minutes)
        
        # Start with normal for a while, then maybe switch to current scenario for the last part
        original_scenario = self.current_scenario
        
        for i in range(minutes * 60 // interval_sec):
            timestamp = base_time + datetime.timedelta(seconds=i * interval_sec)
            
            # For the last 15% of the data, use the current scenario if it's not normal
            if original_scenario != "normal" and i > (minutes * 60 // interval_sec) * 0.85:
                self.set_scenario(original_scenario)
            else:
                self.set_scenario("normal")
                
            metrics = self.get_metrics(device_id)
            metrics["timestamp"] = timestamp
            metrics["device_id"] = device_id
            history.append(metrics)
            
        self.set_scenario(original_scenario)
        return history
