import networkx as nx
from typing import List, Dict

# Physical XY positions on factory floor (in metres — 20x20 grid)
FLOOR_POSITIONS = {
    "core-sw-01":      (10, 18),
    "edge-sw-02":      (10, 12),
    "plc-robot-01":    (3,  4),
    "plc-conveyor-02": (10, 4),
    "hmi-station-03":  (17, 4),
}

# Device type labels for richer display
DEVICE_TYPES = {
    "core-sw-01":      "Core Switch",
    "edge-sw-02":      "Edge Switch",
    "plc-robot-01":    "PLC (Robot)",
    "plc-conveyor-02": "PLC (Conveyor)",
    "hmi-station-03":  "HMI Station",
}

class NetworkTopology:
    def __init__(self):
        self.graph = nx.DiGraph()
        self._build_default_topology()

    def _build_default_topology(self):
        """Builds the topology: core-sw-01 -> edge-sw-02 -> [plc-robot-01, plc-conveyor-02, hmi-station-03]"""
        self.graph.add_edge("core-sw-01", "edge-sw-02")
        self.graph.add_edge("edge-sw-02", "plc-robot-01")
        self.graph.add_edge("edge-sw-02", "plc-conveyor-02")
        self.graph.add_edge("edge-sw-02", "hmi-station-03")

    def get_parent(self, device_id: str) -> str:
        parents = list(self.graph.predecessors(device_id))
        return parents[0] if parents else None

    def get_children(self, device_id: str) -> List[str]:
        return list(self.graph.successors(device_id))

    def trace_upstream(self, device_id: str) -> List[str]:
        path = []
        current = device_id
        while True:
            parent = self.get_parent(current)
            if parent:
                path.append(parent)
                current = parent
            else:
                break
        return path

    def get_impacted_devices(self, root_cause_device: str) -> List[str]:
        """Returns all downstream devices from a faulty node."""
        return list(nx.descendants(self.graph, root_cause_device))

    def get_topology_data(self):
        """Returns nodes and edges for visualization."""
        nodes = [{"id": n, "label": n} for n in self.graph.nodes()]
        edges = [{"from": u, "to": v} for u, v in self.graph.edges()]
        return {"nodes": nodes, "edges": edges}

    def get_health_color(self, score: float) -> str:
        """Returns a color string based on health score."""
        if score >= 80:
            return "#2ecc71"   # green
        elif score >= 50:
            return "#f39c12"   # amber
        else:
            return "#e74c3c"   # red

    def get_topology_plotly_data(self, device_health: Dict[str, float]):
        """
        Returns Plotly-compatible node/edge data for the interactive topology graph.
        device_health: {device_id: one_score (0-100)}
        """
        # Use a fixed hierarchical layout for clarity
        pos = {
            "core-sw-01":      (0.5, 1.0),
            "edge-sw-02":      (0.5, 0.6),
            "plc-robot-01":    (0.1, 0.1),
            "plc-conveyor-02": (0.5, 0.1),
            "hmi-station-03":  (0.9, 0.1),
        }

        edge_traces = []
        for u, v in self.graph.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_traces.append({
                "x": [x0, x1, None],
                "y": [y0, y1, None],
            })

        node_data = []
        for node in self.graph.nodes():
            x, y = pos[node]
            score = device_health.get(node, 100.0)
            color = self.get_health_color(score)
            node_data.append({
                "id": node,
                "label": DEVICE_TYPES.get(node, node),
                "x": x,
                "y": y,
                "score": score,
                "color": color,
            })

        return edge_traces, node_data

    def get_floor_plan_data(self, device_health: Dict[str, float]):
        """
        Returns floor-plan device data with XY positions and health colors.
        device_health: {device_id: one_score (0-100)}
        """
        devices = []
        for device_id, (fx, fy) in FLOOR_POSITIONS.items():
            score = device_health.get(device_id, 100.0)
            color = self.get_health_color(score)
            devices.append({
                "id": device_id,
                "label": DEVICE_TYPES.get(device_id, device_id),
                "x": fx,
                "y": fy,
                "score": score,
                "color": color,
            })

        # Cable run lines (same as graph edges)
        cables = [(u, v) for u, v in self.graph.edges()]
        return devices, cables
