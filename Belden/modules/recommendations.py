from typing import List, Dict

class RecommendationEngine:
    def __init__(self):
        self.actions = {
            "Cable_Failure": {
                "priority": "HIGH",
                "action": "Test cable with TDR (Time Domain Reflectometer)",
                "time": "15 minutes",
                "confirmation": "Replace cable"
            },
            "EMI_Interference": {
                "priority": "MEDIUM",
                "action": "Scan for EMI sources near device",
                "time": "20 minutes",
                "confirmation": "Shield cable or relocate"
            },
            "Config_Error": {
                "priority": "LOW",
                "action": "Check interface configuration (duplex/speed)",
                "time": "5 minutes",
                "confirmation": "Reconfigure interface"
            }
        }

    def generate_plan(self, diagnosis_results: Dict[str, float]) -> List[Dict]:
        """Generates a list of recommended actions sorted by probability."""
        sorted_causes = sorted(diagnosis_results.items(), key=lambda x: x[1], reverse=True)
        
        plan = []
        for i, (cause, prob) in enumerate(sorted_causes):
            if cause in self.actions:
                action_info = self.actions[cause].copy()
                action_info["cause"] = cause
                action_info["probability"] = prob
                action_info["rank"] = i + 1
                plan.append(action_info)
                
        return plan
