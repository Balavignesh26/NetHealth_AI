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

# Initialize Orchestrator in Session State
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = Orchestrator()
    # Initial load
    st.session_state.orchestrator.load_data('data/raw/metrics_timeseries.csv', 'data/raw/assets.json')
    st.session_state.scenario = "Normal"

if 'ai_assistant' not in st.session_state:
    st.session_state.ai_assistant = AIAssistant()

# Sidebar
logo_path = "data/raw/belden-logo.jpeg"
# Check if logo exists locally, otherwise use remote URL as fallback
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=200)
else:
    st.sidebar.warning(f"Logo not found at {logo_path}")
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/4/4e/Belden_Inc_logo.svg", width=150)

st.sidebar.title("Controls")

scenario = st.sidebar.radio("Simulate Scenario", ["Normal Operation", "Inject Fault (Cable Failure)", "Severe Fault (L4 Attack)"])

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
diagnosis = orch.run_diagnosis_pipeline(anomalies)

# Calculate One Score Average
total_score = 0
count = 0
for asset_id, scores in orch.latest_kpis.items():
    total_score += scores['one_score']
    count += 1
avg_score = round(total_score / count, 1) if count > 0 else 100.0

# Render UI
render_top_bar(avg_score, len(anomalies))

col_main, col_right = st.columns([2, 1])

with col_main:
    st.markdown("### Network Map")
    render_topology(orch.topology, anomalies)
    
    st.markdown("### Asset Health")
    render_health_metrics(orch.latest_kpis)

with col_right:
    render_ai_insights(diagnosis)
    
    st.subheader("Active Anomalies")
    if anomalies:
        for a in anomalies:
            st.warning(f"**{a.asset_id}**: {a.description}")
    else:
        st.success("No active anomalies.")

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
