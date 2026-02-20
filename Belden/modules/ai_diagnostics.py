import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from statsmodels.tsa.stattools import grangercausalitytests
from typing import Dict, List, Tuple

class AIDiagnostics:
    def __init__(self):
        self.iso_forest = IsolationForest(contamination=0.1)
        self._init_bayesian_network()

    def _init_bayesian_network(self):
        """Initializes the Bayesian Network structure as per Step 5."""
        # Nodes: Root Cause (RC), Symptoms (S)
        # RC: Cable_Failure, EMI_Interference, Config_Error
        # S: CRC_High, Temp_High, Latency_High, PL_High
        
        self.model = DiscreteBayesianNetwork([
            ('Cable_Failure', 'CRC_High'),
            ('Cable_Failure', 'Temp_High'),
            ('Cable_Failure', 'Latency_High'),
            ('EMI_Interference', 'CRC_High'),
            ('EMI_Interference', 'Latency_High'),
            ('Config_Error', 'Latency_High'),
            ('Config_Error', 'CRC_High')
        ])

        # Define Conditional Probability Distributions (CPDs)
        # Probabilities are based on user's examples and logical inferences
        cpd_cable = TabularCPD(variable='Cable_Failure', variable_card=2, values=[[0.85], [0.15]]) # P(No), P(Yes)
        cpd_emi = TabularCPD(variable='EMI_Interference', variable_card=2, values=[[0.90], [0.10]])
        cpd_config = TabularCPD(variable='Config_Error', variable_card=2, values=[[0.95], [0.05]])

        # S: CRC_High | Cable, EMI, Config
        # Evidence: P(CRC_High=True | Cable, EMI, Config)
        # 2*2*2 = 8 combinations
        cpd_crc = TabularCPD(
            variable='CRC_High', variable_card=2,
            values=[
                [0.99, 0.15, 0.40, 0.05, 0.30, 0.02, 0.10, 0.01], # False
                [0.01, 0.85, 0.60, 0.95, 0.70, 0.98, 0.90, 0.99]  # True
            ],
            evidence=['Cable_Failure', 'EMI_Interference', 'Config_Error'],
            evidence_card=[2, 2, 2]
        )

        cpd_temp = TabularCPD(
            variable='Temp_High', variable_card=2,
            values=[
                [0.98, 0.30], # False
                [0.02, 0.70]  # True (If cable failure, temp is likely high)
            ],
            evidence=['Cable_Failure'],
            evidence_card=[2]
        )

        cpd_latency = TabularCPD(
            variable='Latency_High', variable_card=2,
            values=[
                [0.99, 0.20, 0.50, 0.10, 0.40, 0.05, 0.30, 0.02],
                [0.01, 0.80, 0.50, 0.90, 0.60, 0.95, 0.70, 0.98]
            ],
            evidence=['Cable_Failure', 'EMI_Interference', 'Config_Error'],
            evidence_card=[2, 2, 2]
        )

        self.model.add_cpds(cpd_cable, cpd_emi, cpd_config, cpd_crc, cpd_temp, cpd_latency)
        self.inference = VariableElimination(self.model)

    def detect_anomalies(self, data: pd.DataFrame) -> pd.Series:
        """Trains and detects anomalies using Isolation Forest."""
        # Use metrics for anomaly detection
        metrics_cols = ['crc_errors', 'latency', 'temperature', 'packet_loss', 'retransmits']
        X = data[metrics_cols]
        self.iso_forest.fit(X)
        scores = self.iso_forest.decision_function(X)
        # Anomaly if score is low (Isolation Forest convention)
        return pd.Series(scores).apply(lambda x: 1 if x < -0.05 else 0)

    def diagnose_root_cause(self, symptoms: Dict[str, bool]) -> Dict[str, float]:
        """Runs Bayesian inference to find most likely root cause."""
        # Map raw symptoms to model variables
        evidence = {}
        if 'crc_errors' in symptoms: evidence['CRC_High'] = 1 if symptoms['crc_errors'] else 0
        if 'temperature' in symptoms: evidence['Temp_High'] = 1 if symptoms['temperature'] else 0
        if 'latency' in symptoms: evidence['Latency_High'] = 1 if symptoms['latency'] else 0
        
        results = {}
        for cause in ['Cable_Failure', 'EMI_Interference', 'Config_Error']:
            q = self.inference.query(variables=[cause], evidence=evidence)
            results[cause] = q.values[1] # Probability of True
            
        return results

    def verify_causality(self, series_a: List[float], series_b: List[float]) -> float:
        """Runs Granger Causality test and returns p-value."""
        if len(series_a) < 20: return 1.0 # Not enough data
        
        data = pd.DataFrame({'A': series_a, 'B': series_b})
        # Test if A causes B
        try:
            res = grangercausalitytests(data, maxlag=5, verbose=False)
            # Get p-value for the first lag
            p_val = res[1][0]['ssr_ftest'][1]
            return p_val
        except:
            return 1.0
