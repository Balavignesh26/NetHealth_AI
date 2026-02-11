# System Architecture

## Overview
The Belden ONE View platform is a modular observability system designed for industrial networks.

## Components

### 1. Data Layer (`src/data`)
*   **Schemas**: Pydantic models for strict type checking.
*   **Loader**: Ingests CSV (Metrics) and JSON (Assets).

### 2. Core Engine (`src/core`)
*   **KPI Engine**: Calculates L1/L3/L4 health scores and the aggregated ONE Score.
*   **Topology**: Builds a directed graph of network assets to understand dependencies.

### 3. Intelligence Layer (`src/intelligence`)
*   **Anomaly Detection**: Uses Isolation Forest to detect statistical outliers in time-series data.
*   **Correlator**: Rule-based engine that uses topology to trace root causes (e.g., Upstream Dominance).
*   **Explainer**: Generates natural language summaries for non-technical users.

### 4. Orchestration (`src/orchestration`)
*   **Pipeline**: Connects data ingestion, analysis, and insights generation into a single workflow.

### 5. Presentation (`src/dashboard`)
*   **Streamlit App**: Interactive dashboard for visualization.
*   **Components**: Topology Map, Health Charts, AI Insights Panel.

## Data Flow
1.  **Ingest**: Raw metrics -> `DataLoader` -> `MetricRecord` objects.
2.  **Process**: Metrics -> `KPIEngine` -> Health Scores.
3.  **Detect**: Metrics -> `AnomalyDetector` -> `Anomaly` events.
4.  **Diagnose**: Anomalies + Topology -> `Correlator` -> `RootCause`.
5.  **Explain**: `RootCause` -> `Explainer` -> Text.
6.  **Visualize**: All outputs -> Streamlit Dashboard.
