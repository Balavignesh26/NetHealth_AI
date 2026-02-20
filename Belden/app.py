import streamlit as st
import pandas as pd
import numpy as np
import time
import datetime
import os
from modules.data_source import SyntheticDataSource
from modules.kpi_engine import KPICalculator
from modules.ai_diagnostics import AIDiagnostics
from modules.topology import NetworkTopology
from modules.thermal_model import ThermalPredictionModel
from modules.recommendations import RecommendationEngine
import plotly.express as px
import plotly.graph_objects as go 

# Page Config
st.set_page_config(page_title="Belden AI Network Diagnostics", layout="wide")

# Initialize Modules (Cached if possible, but for prototype we just init)
if 'data_source' not in st.session_state:
    st.session_state.data_source = SyntheticDataSource()
if 'kpi_engine' not in st.session_state:
    st.session_state.kpi_engine = KPICalculator()
if 'ai_diagnostics' not in st.session_state:
    st.session_state.ai_diagnostics = AIDiagnostics()
if 'topology' not in st.session_state:
    st.session_state.topology = NetworkTopology()
if 'thermal_model' not in st.session_state:
    st.session_state.thermal_model = ThermalPredictionModel()
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = RecommendationEngine()

# --- HEADER ---
st.title("NetHealth Prototype")
st.markdown("---")

# --- SIDEBAR: LOGO ---
_logo_path = os.path.join(os.path.dirname(__file__), "belden-logo.jpeg")
try:
    st.sidebar.image(str(_logo_path), use_container_width=True)
except Exception:
    st.sidebar.markdown("**Belden AI Network Diagnostics**")
st.sidebar.markdown("---")
st.sidebar.header("Simulation Controls")
scenario = st.sidebar.selectbox("Current Scenario", ["Normal Operation", "Cable Failure", "EMI Interference"])
if scenario == "Normal Operation":
    st.session_state.data_source.set_scenario("normal")
elif scenario == "Cable Failure":
    st.session_state.data_source.set_scenario("cable_failure")
else:
    st.session_state.data_source.set_scenario("emi_interference")

st.sidebar.markdown("---")
st.sidebar.subheader("AI Assistant")

# --- SIDEBAR: AI ASSISTANT (placed after data is computed, injected via session_state) ---
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "text": "Hello! I am your AI Network Assistant. I can help you analyze anomalies, check system health, or explain root causes. How can I assist you today?"}
    ]

# --- STEP 1 & 2: DATA COLLECTION ---
# Generate real-time data
device_id = "edge-sw-02"
raw_data = st.session_state.data_source.get_metrics(device_id)

# --- STEP 3: KPI SCORES ---
analysis = st.session_state.kpi_engine.get_full_analysis(raw_data)
one_score = analysis["one_score"]

# Count active anomalies: OSI layers with a score below 80
_layer_anomalies = sum(1 for s in analysis["layer_scores"].values() if s < 80)


col1, col2, col3 = st.columns(3)
with col1:
    st.metric("ONE Health Score", f"{one_score:.1f}/100", delta=f"{one_score-85:.1f}" if one_score < 80 else None)
with col2:
    st.metric("Active Anomalies", str(_layer_anomalies))
with col3:
    status_color = "🔴" if analysis["status"] == "CRITICAL" else "🟡" if analysis["status"] == "WARNING" else "🟢"
    st.metric("System Status", f"{status_color} {analysis['status']}")

# --- STEP 7: TOPOLOGY ---
st.subheader("Network Topology & Factory Floor Map")

# Build per-device health scores:
# The monitored device gets the real score; others get a degraded score if scenario is active.
_scenario = st.session_state.data_source.current_scenario
_base_score = one_score
_degraded = max(0, _base_score - 20)

device_health = {
    "core-sw-01":      _base_score * 0.95 if _scenario == "normal" else _degraded * 1.1,
    "edge-sw-02":      _base_score,
    "plc-robot-01":    _base_score * 0.98 if _scenario == "normal" else _degraded * 0.85,
    "plc-conveyor-02": _base_score * 0.97 if _scenario == "normal" else _degraded * 0.90,
    "hmi-station-03":  _base_score * 0.99 if _scenario == "normal" else _degraded * 0.95,
}
# Clamp all scores 0-100
device_health = {k: min(100, max(0, v)) for k, v in device_health.items()}

topo_col, floor_col = st.columns(2)

# ── LEFT: Interactive Topology Graph ─────────────────────────────────
with topo_col:
    st.markdown("**Network Topology Graph**")
    edge_traces, node_data = st.session_state.topology.get_topology_plotly_data(device_health)

    fig_topo = go.Figure()

    # Draw edges first
    for et in edge_traces:
        fig_topo.add_trace(go.Scatter(
            x=et["x"], y=et["y"],
            mode="lines",
            line=dict(color="#555555", width=2),
            hoverinfo="none",
            showlegend=False
        ))

    # Draw nodes
    fig_topo.add_trace(go.Scatter(
        x=[n["x"] for n in node_data],
        y=[n["y"] for n in node_data],
        mode="markers+text",
        marker=dict(
            size=40,
            color=[n["color"] for n in node_data],
            line=dict(color="#ffffff", width=2),
        ),
        text=[n["label"] for n in node_data],
        textposition="bottom center",
        textfont=dict(size=10, color="#ffffff"),
        customdata=[[n["id"], n["score"]] for n in node_data],
        hovertemplate="<b>%{customdata[0]}</b><br>ONE Score: %{customdata[1]:.1f}%<extra></extra>",
        showlegend=False
    ))

    fig_topo.update_layout(
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#1a1a2e",
        font=dict(color="#ffffff"),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.1, 1.1]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.15, 1.15]),
        height=350,
    )
    st.plotly_chart(fig_topo, use_container_width=True)

# ── RIGHT: Factory Floor Map ──────────────────────────────────────────
with floor_col:
    st.markdown("**Factory Floor Map**")
    floor_devices, cable_runs = st.session_state.topology.get_floor_plan_data(device_health)

    fig_floor = go.Figure()

    # Factory boundary
    fig_floor.add_shape(type="rect", x0=0, y0=0, x1=20, y1=20,
                        line=dict(color="#444466", width=2),
                        fillcolor="rgba(20,20,50,0.6)")

    # Zone labels
    zones = [
        dict(x0=0.5, y0=0.5, x1=6.5,  y1=8,  label="Robot Zone",    color="rgba(52,73,94,0.4)"),
        dict(x0=7,   y0=0.5, x1=13,   y1=8,  label="Conveyor Zone",  color="rgba(44,62,80,0.4)"),
        dict(x0=13.5,y0=0.5, x1=19.5, y1=8,  label="Control Room",   color="rgba(39,55,70,0.4)"),
        dict(x0=2,   y0=9,   x1=18,   y1=16, label="Network Core",   color="rgba(30,40,60,0.4)"),
    ]
    for z in zones:
        fig_floor.add_shape(type="rect", x0=z["x0"], y0=z["y0"], x1=z["x1"], y1=z["y1"],
                            line=dict(color="#334466", width=1), fillcolor=z["color"])
        fig_floor.add_annotation(x=(z["x0"]+z["x1"])/2, y=z["y1"]-0.6,
                                 text=z["label"], showarrow=False,
                                 font=dict(size=9, color="#aabbcc"))

    # Cable runs
    floor_pos_map = {d["id"]: (d["x"], d["y"]) for d in floor_devices}
    for u, v in cable_runs:
        if u in floor_pos_map and v in floor_pos_map:
            x0, y0 = floor_pos_map[u]
            x1, y1 = floor_pos_map[v]
            fig_floor.add_trace(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode="lines",
                line=dict(color="#556677", width=2, dash="dot"),
                hoverinfo="none",
                showlegend=False
            ))

    # Device markers
    fig_floor.add_trace(go.Scatter(
        x=[d["x"] for d in floor_devices],
        y=[d["y"] for d in floor_devices],
        mode="markers+text",
        marker=dict(
            size=22,
            color=[d["color"] for d in floor_devices],
            symbol="circle",
            line=dict(color="#ffffff", width=1.5),
        ),
        text=[d["label"] for d in floor_devices],
        textposition="top center",
        textfont=dict(size=9, color="#ddeeff"),
        customdata=[[d["id"], d["score"]] for d in floor_devices],
        hovertemplate="<b>%{customdata[0]}</b><br>ONE Score: %{customdata[1]:.1f}%<extra></extra>",
        showlegend=False
    ))

    # Legend patches (manual)
    for color, label, sy in [("#2ecc71", "Healthy (>=80)", 19.5),
                              ("#f39c12", "Warning (50-79)", 18.8),
                              ("#e74c3c", "Critical (<50)",  18.1)]:
        fig_floor.add_shape(type="rect", x0=14.5, y0=sy-0.25, x1=15.3, y1=sy+0.25,
                            fillcolor=color, line=dict(color=color))
        fig_floor.add_annotation(x=15.6, y=sy, text=label, showarrow=False, xanchor="left",
                                 font=dict(size=8, color="#ccddee"))

    fig_floor.update_layout(
        paper_bgcolor="#0d0d1a",
        plot_bgcolor="#0d0d1a",
        font=dict(color="#ffffff"),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 20.5]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[-0.5, 20.5], scaleanchor="x", scaleratio=1),
        height=350,
    )
    st.plotly_chart(fig_floor, use_container_width=True)

# --- STEP 4 & 5: ANOMALY & DIAGNOSIS ---
st.markdown("---")
left_col, right_col = st.columns(2)

with left_col:
    st.subheader("AI Root Cause Diagnosis")
    # Determine symptoms for Bayesian Network
    symptoms = {
        "crc_errors": analysis["normalized_metrics"]["crc_errors"] < 50,
        "temperature": analysis["normalized_metrics"]["temperature"] < 50,
        "latency": analysis["normalized_metrics"]["latency"] < 50
    }
    
    diagnosis = st.session_state.ai_diagnostics.diagnose_root_cause(symptoms)
    
    # Plot Diagnosis Probabilities
    fig = px.bar(
        x=list(diagnosis.keys()), 
        y=list(diagnosis.values()),
        labels={'x': 'Root Cause', 'y': 'Probability'},
        title="Bayesian Probabilistic Inference",
        color=list(diagnosis.values()),
        color_continuous_scale="RdYlGn_r"
    )
    st.plotly_chart(fig, use_container_width=True)

with right_col:
    st.subheader("Causal Chain Verification")
    # Generate some historical data for Granger test
    history = st.session_state.data_source.generate_time_series(device_id, minutes=30)
    df_history = pd.DataFrame(history)
    
    # Granger Test: CRC -> Latency
    p_val = st.session_state.ai_diagnostics.verify_causality(df_history['crc_errors'].tolist(), df_history['latency'].tolist())
    
    st.write(f"**Test Setup:** Does CRC Errors Granger-cause Latency?")
    if p_val < 0.05:
        st.success(f"Statistical Proof: CRC DOES Granger-cause Latency (p={p_val:.4f})")
    else:
        st.warning(f"Correlation only (p={p_val:.4f})")
        
    # Plot historical trend
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(y=df_history['crc_errors'], name="CRC Errors", yaxis="y1"))
    fig_trend.add_trace(go.Scatter(y=df_history['latency'], name="Latency (ms)", yaxis="y2"))
    fig_trend.update_layout(
        yaxis=dict(title="CRC Errors"),
        yaxis2=dict(title="Latency", overlaying="y", side="right"),
        title="Cross-Layer Correlation (Last 30 min)"
    )
    st.plotly_chart(fig_trend, use_container_width=True)

# --- STEP 8: THERMAL PREDICTION ---
st.markdown("---")
st.subheader("Thermal Prediction")
thermal_history = st.session_state.thermal_model.simulate_90_days()
summary = st.session_state.thermal_model.get_forecast_summary(thermal_history)

t_col1, t_col2 = st.columns([1, 2])
with t_col1:
    st.write(f"**Current Temperature:** {summary['current_temp']:.1f}°C")
    if summary['failure_risk']:
        st.error(f"FAIL RISK: Cable predicted to fail in {summary['days_to_failure']} days")
        st.warning(f"Action Required: Schedule replacement by Day {summary['recommended_maintenance_day']}")
    else:
        st.success("Thermal health within safe parameters for next 90 days.")

with t_col2:
    df_thermal = pd.DataFrame(thermal_history)
    fig_thermal = px.line(df_thermal, x="day", y="temperature", title="90-Day Temperature Forecast")
    fig_thermal.add_hline(y=75, line_dash="dash", line_color="red", annotation_text="Failure Threshold")
    st.plotly_chart(fig_thermal, use_container_width=True)

# --- STEP 9: RECOMMENDATIONS ---
st.markdown("---")
st.subheader("Multi-Hypothesis Action Plan")
plan = st.session_state.recommendations.generate_plan(diagnosis)

for item in plan:
    with st.expander(f"Priority {item['rank']}: {item['cause'].replace('_', ' ')} ({item['probability']*100:.1f}%)"):
        st.write(f"**Primary Action:** {item['action']}")
        st.write(f"**Estimated Time:** {item['time']}")
        st.write(f"**If Confirmed:** {item['confirmation']}")

st.markdown("---")
st.caption("v1.3")


def _ai_respond(user_msg: str) -> str:
    """Rule-based AI assistant that answers using live diagnosis data."""
    msg = user_msg.lower().strip()

    # Build live context
    score     = one_score
    status    = analysis["status"]
    layers    = analysis["layer_scores"]
    anomalies = _layer_anomalies
    top_cause = max(diagnosis, key=diagnosis.get)
    top_prob  = diagnosis[top_cause] * 100
    thermal_ok = not summary["failure_risk"]
    top_rec   = plan[0] if plan else None

    layer_str = ", ".join(f"{k}:{v:.1f}" for k, v in layers.items())

    if any(k in msg for k in ["health", "score", "status", "overall"]):
        return (
            f"**System Health:** ONE Score is **{score:.1f}/100** — status is **{status}**.\n\n"
            f"Layer breakdown: {layer_str}.\n\n"
            f"There are currently **{anomalies}** layer(s) flagged as anomalous (score < 80)."
        )

    elif any(k in msg for k in ["root cause", "cause", "why", "fault", "failure", "reason"]):
        cause_name = top_cause.replace("_", " ")
        return (
            f"**Root Cause Analysis:** The most likely cause is **{cause_name}** "
            f"with a Bayesian probability of **{top_prob:.1f}%**.\n\n"
            f"All probabilities: " +
            ", ".join(f"{k.replace('_',' ')}: {v*100:.1f}%" for k, v in diagnosis.items()) + "."
        )

    elif any(k in msg for k in ["anomal", "alert", "flag", "issue", "problem"]):
        if anomalies == 0:
            return "No active anomalies detected. All OSI layer scores are within healthy thresholds (>=80)."
        bad = [f"{k} ({v:.1f})" for k, v in layers.items() if v < 80]
        return (
            f"**{anomalies} active anomalie(s)** detected.\n\n"
            f"Affected layers: **{', '.join(bad)}**.\n\n"
            f"Highest-risk cause: **{top_cause.replace('_',' ')}** ({top_prob:.1f}% probability)."
        )

    elif any(k in msg for k in ["thermal", "temp", "heat", "cable", "overh"]):
        if thermal_ok:
            return (
                f"Thermal status is **safe**. Current temperature: **{summary['current_temp']:.1f}°C**. "
                f"No cable failure predicted in the next 90 days."
            )
        else:
            return (
                f"**Thermal ALERT!** Current temperature: **{summary['current_temp']:.1f}°C**.\n\n"
                f"Cable failure predicted in **{summary['days_to_failure']} days**. "
                f"Schedule maintenance by Day {summary['recommended_maintenance_day']}."
            )

    elif any(k in msg for k in ["recommend", "action", "fix", "what should", "next step", "plan"]):
        if top_rec:
            return (
                f"**Top Recommendation (Priority {top_rec['rank']}):** "
                f"Address **{top_rec['cause'].replace('_',' ')}** ({top_rec['probability']*100:.1f}% likely).\n\n"
                f"Action: {top_rec['action']}\n\nEstimated time: {top_rec['time']}\n\n"
                f"If confirmed: {top_rec['confirmation']}"
            )
        return "No recommendations available at this time."

    elif any(k in msg for k in ["layer", "l1", "l3", "l4", "osi"]):
        return (
            f"**OSI Layer Health Scores:**\n\n" +
            "\n\n".join(
                f"- **{k}**: {v:.1f}% — {'OK' if v >= 80 else 'WARNING' if v >= 50 else 'CRITICAL'}"
                for k, v in layers.items()
            )
        )

    elif any(k in msg for k in ["hi", "hello", "hey", "help"]):
        return (
            "Hello! I'm your AI Network Assistant. You can ask me about:\n\n"
            "- **System health** or ONE score\n"
            "- **Root cause** analysis\n"
            "- **Active anomalies**\n"
            "- **Thermal** / cable risk\n"
            "- **Recommendations** / action plan\n"
            "- **Layer** scores (L1, L3, L4...)"
        )

    else:
        return (
            "I'm analyzing the network telemetry. You can ask me about "
            "**system health**, **active anomalies**, **root causes**, "
            "**thermal risk**, or **recommendations**."
        )

# ── Chat display ───────────────────────────────────────────────────────
chat_container = st.sidebar.container()
with chat_container:
    for msg in st.session_state.chat_history:
        if msg["role"] == "assistant":
            st.sidebar.markdown(
                f"<div style='background:#1e2a3a;border-radius:8px;padding:10px 12px;"
                f"margin-bottom:8px;font-size:13px;color:#ddeeff;'>"
                f"<b style='color:#f4a261;'>Assistant</b><br>{msg['text']}</div>",
                unsafe_allow_html=True
            )
        else:
            st.sidebar.markdown(
                f"<div style='background:#2a1e3a;border-radius:8px;padding:8px 12px;"
                f"margin-bottom:8px;font-size:13px;color:#eeddff;text-align:right;'>"
                f"<b style='color:#a29bfe;'>You</b><br>{msg['text']}</div>",
                unsafe_allow_html=True
            )

# ── Input box ──────────────────────────────────────────────────────────
user_input = st.sidebar.text_input(
    label="Ask the assistant",
    placeholder="Ask about network health...",
    label_visibility="collapsed",
    key="ai_chat_input"
)
if st.sidebar.button("Send", use_container_width=True) and user_input.strip():
    st.session_state.chat_history.append({"role": "user", "text": user_input.strip()})
    reply = _ai_respond(user_input.strip())
    st.session_state.chat_history.append({"role": "assistant", "text": reply})
    st.rerun()







