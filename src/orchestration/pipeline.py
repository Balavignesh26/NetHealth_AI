import pandas as pd
from typing import List, Dict, Any
from datetime import datetime

from src.data.loader import load_metrics, load_assets
from src.data.schemas import Anomaly, RootCause, Asset, MetricRecord
from src.core.topology.asset_inventory import AssetInventory
from src.core.topology.topology_builder import TopologyBuilder
from src.core.kpi_engine.one_score import OneScoreCalculator
from src.intelligence.anomaly_detector import AnomalyDetector
from src.intelligence.correlator import Correlator
from src.intelligence.explainer import Explainer
from src.intelligence.predictor import Predictor

class Orchestrator:
    def __init__(self):
        self.assets = []
        self.metrics = []
        self.inventory = None
        self.topology = None
        self.one_score_calc = OneScoreCalculator()
        self.anomaly_detector = AnomalyDetector()
        self.correlator = None
        self.explainer = Explainer()
        self.predictor = Predictor()
        self.latest_kpis = {}
        self.latest_predictions = {}

    def load_data(self, metrics_file: str, assets_file: str):
        self.metrics = load_metrics(metrics_file)
        self.assets = load_assets(assets_file)
        
        self.inventory = AssetInventory(self.assets)
        self.topology = TopologyBuilder(self.assets)
        self.correlator = Correlator(self.topology)
        
        # Train anomaly detector on loaded metrics (simulating history)
        # Convert metrics to DataFrame for training
        if self.metrics:
            df = self._metrics_to_df(self.metrics)
            # Train on numerical columns
            self.anomaly_detector.train(df, ['value']) # Simplified to just value for now
            
    def _metrics_to_df(self, metrics: List[MetricRecord]) -> pd.DataFrame:
        data = [m.model_dump() for m in metrics]
        return pd.DataFrame(data)

    def run_kpi_pipeline(self) -> List[Anomaly]:
        """
        Compute KPIs and Detect Anomalies.
        """
        anomalies = []
        
        # Group metrics by asset and timestamp for OneScore
        # This is a bit complex, simplifying for MVP:
        # Just check each metric for detection and compute OneScore for latest state.
        
        df = self._metrics_to_df(self.metrics)
        
        # Detect Anomalies on raw metrics
        # (Real system would do this on KPIs too)
        # Including L7 metrics in detection candidates
        results = self.anomaly_detector.detect(df, ['value'])
        
        for idx, row in results.iterrows():
            if row['is_anomaly']:
                # Create Anomaly object
                # severity based on score or value
                severity = "high" if row['anomaly_score'] < -0.2 else "medium"
                
                anomaly = Anomaly(
                    id=f"evt_{idx}",
                    timestamp=row['timestamp'],
                    asset_id=row['asset_id'],
                    metric_or_kpi=row['metric_name'],
                    severity=severity,
                    description=f"Anomaly detected in {row['metric_name']}",
                    score=float(row['anomaly_score'])
                )
                anomalies.append(anomaly)
                
        # Also compute ONE Score for each asset (based on latest metrics)
        # Group by asset
        asset_groups = df.groupby('asset_id')
        for asset_id, group in asset_groups:
            # Take latest values for each metric type
            latest_metrics = group.sort_values('timestamp').groupby('metric_name').last()['value'].to_dict()
            scores = self.one_score_calc.calculate_one_score(latest_metrics)
            self.latest_kpis[asset_id] = scores
            
            # If ONE score is low, could also flag anomaly
            if scores['one_score'] < 60:
                 anomalies.append(Anomaly(
                    id=f"kpi_{asset_id}",
                    timestamp=datetime.now(),
                    asset_id=asset_id,
                    metric_or_kpi="ONE_SCORE",
                    severity="critical",
                    description=f"Health Score Critical: {scores['one_score']} (L1:{scores['l1_score']}, L3:{scores['l3_score']}, L4:{scores['l4_score']}, L7:{scores['l7_score']})",
                    score=scores['one_score']
                ))

        # Generate Predictions for critical metrics (Latency, Throughput, etc)
        # Simplified: Predict for latency and cpu_usage for all assets
        prediction_metrics = ['latency', 'cpu_usage', 'throughput']
        
        for asset_id in self.latest_kpis.keys():
            asset_preds = {}
            for metric in prediction_metrics:
                # Get history from main df
                # Ideally we'd optimize this access
                pass # Logic moved to Predictor class which needs DF.
                
                # Let's call predictor with the full DF relative to this asset
                # Filter DF for this asset
                # This could be slow in prod, ok for hackathon
                
                forecast = self.predictor.forecast_asset_metrics(df, asset_id, metric)
                if forecast:
                    asset_preds[metric] = forecast
            
            if asset_preds:
                self.latest_predictions[asset_id] = asset_preds

        return anomalies

    def run_diagnosis_pipeline(self, anomalies: List[Anomaly]) -> List[Dict[str, Any]]:
        """
        Correlate and Explain.
        """
        root_causes = self.correlator.correlate(anomalies)
        
        results = []
        for rc in root_causes:
            explanation = self.explainer.explain(rc)
            results.append({
                "root_cause": rc,
                "explanation": explanation
            })
            
        return results
