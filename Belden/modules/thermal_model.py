import math
import numpy as np
from typing import List, Dict

class ThermalPredictionModel:
    def __init__(self, ambient_temp: float = 35.0, current_i: float = 5.0):
        self.ambient_temp = ambient_temp
        self.current_i = current_i
        self.R0 = 0.5 # Baseline resistance at 25C
        self.k_cooling = 0.1 # Cooling coefficient
        self.heat_capacity = 2.0 # Heat capacity C
        
    def calculate_resistance(self, temp: float) -> float:
        """R(T) = R0 * (1 + 0.004 * (T - 25))"""
        return self.R0 * (1 + 0.004 * (temp - 25))

    def calculate_power(self, temp: float) -> float:
        """P = I^2 * R(T)"""
        return (self.current_i ** 2) * self.calculate_resistance(temp)

    def simulate_90_days(self, cable_age_days: int = 0) -> List[Dict]:
        """Solves dT/dt = (P - k*deltaT) / C over 90 days.
        Simplified for presentation: we simulate daily steps.
        """
        history = []
        current_temp = self.ambient_temp
        
        # We simulate aging by increasing current_i slightly or decreasing cooling
        # In this simulation, resistance increases slightly with age
        age_factor = 1 + (cable_age_days / 365) * 0.1
        
        for day in range(90):
            # Joule Heating
            P = self.calculate_power(current_temp) * age_factor
            delta_t = current_temp - self.ambient_temp
            
            # dT/dt
            dT_dt = (P - self.k_cooling * delta_t) / self.heat_capacity
            
            # Update temp for next day (simplified Euler)
            current_temp += dT_dt * 0.1 # Scaled time step
            
            # Compute SNR and BER (Step 8)
            snr = 25 - (current_temp - 35) * 0.4
            ber = 10 ** (-(snr/2)) # Rough estimation
            
            status = "Healthy" if current_temp < 50 else "Warning" if current_temp < 70 else "FAIL"
            
            history.append({
                "day": day,
                "temperature": current_temp,
                "snr": snr,
                "ber": ber,
                "status": status
            })
            
            # Add a bit of daily aging
            age_factor += 0.002
            
        return history

    def get_forecast_summary(self, history: List[Dict]) -> Dict:
        failure_day = next((h["day"] for h in history if h["status"] == "FAIL"), None)
        warning_day = next((h["day"] for h in history if h["status"] == "Warning"), None)
        
        return {
            "current_temp": history[0]["temperature"],
            "failure_risk": True if failure_day else False,
            "days_to_failure": failure_day,
            "days_to_warning": warning_day,
            "recommended_maintenance_day": warning_day - 5 if warning_day else None
        }
