from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field

class MetricRecord(BaseModel):
    timestamp: datetime
    asset_id: str
    metric_name: str
    value: float
    unit: Optional[str] = None

class Asset(BaseModel):
    id: str
    name: str
    type: str  # e.g., 'switch', 'plc', 'sensor'
    role: Optional[str] = None
    parent_id: Optional[str] = None # For topology
    metadata: Dict[str, Any] = Field(default_factory=dict)

class KPIRecord(BaseModel):
    timestamp: datetime
    asset_id: str
    kpi_name: str
    value: float
    baseline_mean: Optional[float] = None
    baseline_std: Optional[float] = None
    is_anomaly: bool = False

class Anomaly(BaseModel):
    id: str
    timestamp: datetime
    asset_id: str
    metric_or_kpi: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    score: float

class RootCause(BaseModel):
    anomaly_id: str
    root_cause_asset_id: str
    probability: float
    description: str
    recommended_action: str
