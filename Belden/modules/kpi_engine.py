import math
import numpy as np
from typing import Dict, List

class KPICalculator:
    def __init__(self):
        # Thresholds and k-values for sigmoid normalization
        # Score = 100 / (1 + e^(k * (value - threshold)))
        self.config = {
            "crc_errors": {"threshold": 10, "k": 0.05, "higher_is_worse": True},
            "snr": {"threshold": 20, "k": -0.5, "higher_is_worse": False},
            "latency": {"threshold": 100, "k": 0.02, "higher_is_worse": True},
            "packet_loss": {"threshold": 1, "k": 2.0, "higher_is_worse": True},
            "temperature": {"threshold": 75, "k": 0.2, "higher_is_worse": True},
            "cpu_load": {"threshold": 70, "k": 0.1, "higher_is_worse": True},
            "retransmits": {"threshold": 5, "k": 0.5, "higher_is_worse": True},
        }
        
        # Layer Weight Definitions
        self.layer_definitions = {
            "L1": {"crc_errors": 0.6, "snr": 0.4},
            "L3": {"latency": 0.7, "packet_loss": 0.3},
            "L4": {"retransmits": 1.0},
            "L5_7": {"cpu_load": 0.5, "temperature": 0.5} # Mocked aggregation
        }
        
        # Final ONE Score weights
        self.one_weights = {
            "L1": 0.20,
            "L2": 0.15, # Layer 2 mocked as healthy (98%) for now
            "L3": 0.20,
            "L4": 0.20,
            "L5_7": 0.25
        }

    def sigmoid_normalize(self, value: float, metric_name: str) -> float:
        if metric_name not in self.config:
            return 100.0 # Default to healthy for unknown metrics
        
        cfg = self.config[metric_name]
        k = cfg["k"]
        threshold = cfg["threshold"]
        
        # For 'higher is worse' metrics, score decreases as value increases beyond threshold
        # For 'higher is better' (like SNR), k is negative in my config to handle it
        score = 100 / (1 + math.exp(k * (value - threshold)))
        return max(0.0, min(100.0, score))

    def compute_layer_scores(self, raw_metrics: Dict[str, float]) -> Dict[str, float]:
        layer_scores = {}
        
        # Compute normalized scores for all metrics first
        norm_scores = {name: self.sigmoid_normalize(val, name) for name, val in raw_metrics.items()}
        
        for layer, metrics in self.layer_definitions.items():
            score = 0
            for metric, weight in metrics.items():
                score += norm_scores.get(metric, 100.0) * weight
            layer_scores[layer] = score
            
        # Mock L2 as 98%
        layer_scores["L2"] = 98.0
        
        return layer_scores

    def compute_one_score(self, layer_scores: Dict[str, float]) -> float:
        one_score = 0
        for layer, weight in self.one_weights.items():
            one_score += layer_scores.get(layer, 100.0) * weight
        return one_score

    def get_full_analysis(self, raw_metrics: Dict[str, float]) -> Dict:
        layer_scores = self.compute_layer_scores(raw_metrics)
        one_score = self.compute_one_score(layer_scores)
        
        return {
            "raw_metrics": raw_metrics,
            "normalized_metrics": {name: self.sigmoid_normalize(val, name) for name, val in raw_metrics.items()},
            "layer_scores": layer_scores,
            "one_score": one_score,
            "status": "CRITICAL" if one_score < 50 else "WARNING" if one_score < 75 else "HEALTHY"
        }
