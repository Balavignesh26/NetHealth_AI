from typing import List, Dict, Tuple
from src.data.schemas import Anomaly, RootCause, Asset
from src.core.topology.topology_builder import TopologyBuilder
import networkx as nx

class Correlator:
    def __init__(self, topology: TopologyBuilder):
        self.topology = topology

    def correlate(self, anomalies: List[Anomaly]) -> List[RootCause]:
        """
        Correlate anomalies to find root causes.
        Simplification: We assume provided anomalies are from the same time window.
        """
        if not anomalies:
            return []

        # Map asset_id to anomalies
        anomaly_map = {a.asset_id: a for a in anomalies}
        affected_assets = set(anomaly_map.keys())
        
        root_causes = []
        
        # Simple algorithm: Finding the "most upstream" anomalous node in the graph subgraph induced by anomalies.
        # But we also need to respect Layer precedence?
        # Let's use Upstream Dominance first.
        
        # 1. Identify "Source" anomalies (nodes that have no anomalous ancestors)
        for asset_id in affected_assets:
            ancestors = set(nx.ancestors(self.topology.graph, asset_id))
            anomalous_ancestors = ancestors.intersection(affected_assets)
            
            if not anomalous_ancestors:
                # This node is a root cause candidate
                # (No upstream anomalies found among the set of current anomalies)
                anomaly = anomaly_map[asset_id]
                
                # Check metrics for explanation
                description = f"Root cause identified at {asset_id}. "
                if "crc" in anomaly.metric_or_kpi.lower():
                    description += "Physical layer issue (CRC Errors) indicating cable/EMI problem."
                    action = "Check physical cabling and shielding."
                    prob = 0.95
                elif "loss" in anomaly.metric_or_kpi.lower():
                    description += "Network layer issue (Packet Loss). Check upstream congestion or link."
                    action = "Investigate switch buffers and link capacity."
                    prob = 0.8
                else: 
                    description += f"Anomaly in {anomaly.metric_or_kpi}."
                    action = "General inspection required."
                    prob = 0.7
                
                rc = RootCause(
                    anomaly_id=anomaly.id,
                    root_cause_asset_id=asset_id,
                    probability=prob,
                    description=description,
                    recommended_action=action
                )
                root_causes.append(rc)
                
        return root_causes
