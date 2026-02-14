import streamlit as st
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from src.orchestration.pipeline import Orchestrator
from src.dashboard.components.top_bar import render_top_bar
from src.dashboard.components.topology_view import render_topology
from src.dashboard.components.ai_insights import render_ai_insights
from src.dashboard.components.health_metrics import render_health_metrics
from src.intelligence.ai_assistant import AIAssistant
from src.dashboard.components.chat_interface import ChatInterface
from src.dashboard.components.thermal_view import render_thermal_view
from src.dashboard.components.validation_metrics import render_validation_metrics
from src.dashboard.components.floor_plan_view import render_floor_plan
from src.dashboard.components.security_view import render_security_dashboard
from src.dashboard.data_source import SyntheticDataSource, DatabaseDataSource
from src.dashboard.components.collector_status import render_collector_status_sidebar, render_collector_management

st.set_page_config(page_title="Belden ONE View", layout="wide")

# Custom CSS for "TV Frame" effect
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
    }
    .graphviz_chart {
        border: 2px solid #333;
        border-radius: 10px;
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.2);
        padding: 10px;
        background-color: black;
    }
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
        color: #E0E0E0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Data Source
if 'data_source_mode' not in st.session_state:
    st.session_state.data_source_mode = "Synthetic (Demo)"

if 'data_source' not in st.session_state:
    st.session_state.data_source = SyntheticDataSource()

# Initialize Database Manager for production mode
if 'db_manager' not in st.session_state:
    st.session_state.db_manager = None

# Initialize Orchestrator in Session State
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = Orchestrator()
    # Initial load
    st.session_state.orchestrator.load_data('data/raw/metrics_timeseries.csv', 'data/raw/assets.json')
    st.session_state.scenario = "Normal"

if 'ai_assistant' not in st.session_state:
    st.session_state.ai_assistant = AIAssistant()

# Initialize Intelligence Orchestrator (Phase 2)
if 'intelligence_orchestrator' not in st.session_state:
    try:
        from src.intelligence.orchestrator import IntelligenceOrchestrator
        st.session_state.intelligence_orchestrator = IntelligenceOrchestrator(use_deep_learning=False)
    except Exception as e:
        print(f"Intelligence orchestrator not available: {e}")
        st.session_state.intelligence_orchestrator = None

# Sidebar
logo_path = "data/raw/belden-logo.jpeg"
# Check if logo exists locally, otherwise use remote URL as fallback
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=200)
else:
    st.sidebar.warning(f"Logo not found at {logo_path}")
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/4/4e/Belden_Inc_logo.svg", width=150)

st.sidebar.title("Controls")

# Data Source Selection
st.sidebar.markdown("### 📊 Data Source")
data_source_mode = st.sidebar.radio(
    "Select Data Source",
    ["Synthetic (Demo)", "Production (Live)"],
    help="Synthetic: Use demo CSV data | Production: Use live TimescaleDB data"
)

# Handle data source switching
if data_source_mode != st.session_state.data_source_mode:
    st.session_state.data_source_mode = data_source_mode
    
    if data_source_mode == "Production (Live)":
        # Initialize database connection
        try:
            from src.database import init_database, get_db_manager
            
            if st.session_state.db_manager is None:
                # Try to connect to database
                try:
                    st.session_state.db_manager = get_db_manager()
                    if not st.session_state.db_manager.health_check():
                        st.sidebar.error("❌ Database connection failed!")
                        st.session_state.data_source_mode = "Synthetic (Demo)"
                        st.session_state.data_source = SyntheticDataSource()
                    else:
                        st.session_state.data_source = DatabaseDataSource(st.session_state.db_manager)
                        st.sidebar.success("✅ Connected to production database")
                except Exception as e:
                    st.sidebar.error(f"❌ Database error: {e}")
                    st.session_state.data_source_mode = "Synthetic (Demo)"
                    st.session_state.data_source = SyntheticDataSource()
            else:
                st.session_state.data_source = DatabaseDataSource(st.session_state.db_manager)
        except ImportError:
            st.sidebar.warning("⚠️ Database module not available. Using synthetic data.")
            st.session_state.data_source_mode = "Synthetic (Demo)"
            st.session_state.data_source = SyntheticDataSource()
    else:
        # Switch to synthetic data
        st.session_state.data_source = SyntheticDataSource()

# Show collector status if in production mode
if st.session_state.data_source_mode == "Production (Live)" and st.session_state.db_manager:
    render_collector_status_sidebar(st.session_state.db_manager)

st.sidebar.markdown("---")

# Scenario selection (only for synthetic mode)
if st.session_state.data_source_mode == "Synthetic (Demo)":
    st.sidebar.markdown("### 🎭 Scenario Simulation")
    scenario = st.sidebar.radio("Simulate Scenario", ["Normal Operation", "Inject Fault (Cable Failure)", "Severe Fault (L4 Attack)"])
else:
    scenario = "Production"  # No scenario simulation in production mode

if st.sidebar.button("Run Analysis"):
    # Reload data based on scenario
    orch = st.session_state.orchestrator
    
    if "Normal" in scenario:
        orch.load_data('data/raw/metrics_timeseries.csv', 'data/raw/assets.json')
        st.session_state.scenario = "Normal"
    elif "Cable" in scenario:
        orch.load_data('data/raw/metrics_faulty.csv', 'data/raw/assets.json')
        st.session_state.scenario = "Faulty"
    elif "Severe" in scenario:
        orch.load_data('data/raw/metrics_severe.csv', 'data/raw/assets.json')
        st.session_state.scenario = "Severe"
        
    st.rerun()

# Run Pipeline
orch = st.session_state.orchestrator
anomalies = orch.run_kpi_pipeline()

# Run Thermal Simulation Pipeline
thermal_predictions = orch.run_thermal_simulation_pipeline()

# Correlate thermal predictions with anomalies
anomalies = orch.correlate_thermal_with_anomalies(anomalies)

# Run Granger Causality Analysis (if sufficient data)
if 'causal_graph' not in st.session_state or st.session_state.get('rebuild_causal', False):
    try:
        st.session_state.causal_graph = orch.run_causality_analysis_pipeline()
        st.session_state.rebuild_causal = False
    except Exception as e:
        st.session_state.causal_graph = None
        print(f"Causality analysis failed: {e}")

diagnosis = orch.run_diagnosis_pipeline(anomalies)

# Try Bayesian diagnosis if available
bayesian_diagnosis = None
try:
    from src.intelligence.bayesian_diagnostics import ProbabilisticDiagnosticEngine
    
    if 'bayesian_engine' not in st.session_state:
        st.session_state.bayesian_engine = ProbabilisticDiagnosticEngine()
    
    # Convert anomalies to Bayesian evidence
    if anomalies:
        # Simple mapping - in production would be more sophisticated
        evidence = {}
        for a in anomalies:
            if 'crc' in a.metric_or_kpi.lower():
                evidence['CRCErrors'] = 'High' if a.severity in ['high', 'critical'] else 'Medium'
            elif 'packet' in a.metric_or_kpi.lower():
                evidence['PacketLoss'] = 'High' if a.severity in ['high', 'critical'] else 'Medium'
            elif 'latency' in a.metric_or_kpi.lower():
                evidence['Latency'] = 'VeryHigh' if a.severity == 'critical' else 'High'
        
        if evidence:
            bayesian_diagnosis = st.session_state.bayesian_engine.diagnose_with_uncertainty(evidence)
except Exception as e:
    print(f"Bayesian diagnosis not available: {e}")
    bayesian_diagnosis = None

# Calculate One Score Average
total_score = 0
count = 0
for asset_id, scores in orch.latest_kpis.items():
    total_score += scores['one_score']
    count += 1
avg_score = round(total_score / count, 1) if count > 0 else 100.0

# Render UI
render_top_bar(avg_score, len(anomalies))

# Create tabs for different views
if st.session_state.data_source_mode == "Production (Live)":
    tabs = st.tabs(["🗺️ Network Map", "🏭 Floor Plan", "🌡️ Thermal Twin", "📊 System Performance", "🔒 Security", "📡 Collectors"])
else:
    tabs = st.tabs(["🗺️ Network Map", "🏭 Floor Plan", "🌡️ Thermal Twin", "📊 System Performance", "🔒 Security"])

with tabs[0]:  # Network Map
    col_main, col_right = st.columns([2, 1])
    
    with col_main:
        st.markdown("### Network Topology")
        render_topology(orch.topology, anomalies)
        
        st.markdown("### Asset Health Metrics")
        render_health_metrics(orch.latest_kpis)
    
    with col_right:
        # Enhanced AI insights with Bayesian and Granger causality
        render_ai_insights(
            diagnosis,
            bayesian_diagnosis=bayesian_diagnosis,
            causal_graph=st.session_state.get('causal_graph')
        )
        
        st.subheader("Active Anomalies")
        if anomalies:
            for a in anomalies:
                st.warning(f"**{a.asset_id}**: {a.description}")
        else:
            st.success("No active anomalies.")

with tabs[1]:  # Floor Plan
    st.markdown("## 🏭 Factory Floor Plan - Spatial Health View")
    st.caption("Interactive device positioning with health-based heatmap overlay")
    render_floor_plan(
        assets=orch.assets,
        kpis=orch.latest_kpis,
        anomalies=anomalies,
        floor_plan_path='data/raw/factory_floor_plan.png'
    )

with tabs[2]:  # Thermal Twin
    st.markdown("## 🌡️ Thermal Network Digital Twin")
    st.caption("Physics-based failure prediction using thermal dynamics")
    render_thermal_view(orch.latest_thermal_predictions, orch.assets)

with tabs[3]:  # System Performance
    render_validation_metrics()

with tabs[4]:  # Security
    if st.session_state.intelligence_orchestrator:
        render_security_dashboard(st.session_state.intelligence_orchestrator)
    else:
        st.info("Security monitoring requires Intelligence Orchestrator. Enable deep learning features to access security dashboard.")

# Collectors tab (only in production mode)
if st.session_state.data_source_mode == "Production (Live)":
    with tabs[5]:  # Collectors
        if st.session_state.db_manager:
            render_collector_management(st.session_state.db_manager)
        else:
            st.error("Database connection required for collector management")

# Update AI Context
st.session_state.ai_assistant.update_context(
    anomalies=anomalies,
    kpis=orch.latest_kpis,
    topology=orch.topology,
    predictions=orch.latest_predictions
)

# Render Chat Interface
chat_interface = ChatInterface(st.session_state.ai_assistant)
chat_interface.render()
