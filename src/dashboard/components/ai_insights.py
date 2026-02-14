import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional


def render_ai_insights(
    diagnosis_results: List[Dict[str, Any]],
    bayesian_diagnosis: Optional[Any] = None,
    causal_graph: Optional[Any] = None
):
    """
    Render AI insights with enhanced Bayesian and causality visualizations.
    
    Args:
        diagnosis_results: Traditional root cause analysis results
        bayesian_diagnosis: ProbabilisticDiagnosis object (if available)
        causal_graph: CausalGraph object (if available)
    """
    st.subheader("🤖 AI Root Cause Analysis")
    
    # Bayesian Probabilistic Diagnosis
    if bayesian_diagnosis:
        render_bayesian_diagnosis(bayesian_diagnosis)
        st.markdown("---")
    
    # Traditional Root Cause Analysis
    if not diagnosis_results:
        st.info("No active root causes identified. System looks healthy!")
    else:
        render_traditional_diagnosis(diagnosis_results)
    
    # Granger Causality Proof
    if causal_graph:
        st.markdown("---")
        render_granger_causality(causal_graph)


def render_bayesian_diagnosis(diagnosis):
    """Render Bayesian probabilistic diagnosis with visualization"""
    st.markdown("### 🎲 Probabilistic Diagnosis (Bayesian Network)")
    
    st.markdown(f"""
    **Confidence Level**: {diagnosis.confidence_level}  
    **Primary Hypothesis**: {diagnosis.primary_cause.replace('_', ' ')} ({diagnosis.primary_probability:.1%})
    """)
    
    # Probability Distribution Bar Chart
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create bar chart
        causes = list(diagnosis.cause_probabilities.keys())
        probabilities = list(diagnosis.cause_probabilities.values())
        
        # Format labels
        formatted_causes = [c.replace('_', ' ').title() for c in causes]
        
        # Create figure
        fig = go.Figure()
        
        # Add bars with color coding
        colors = ['#FF6B6B' if p == max(probabilities) else '#4ECDC4' for p in probabilities]
        
        fig.add_trace(go.Bar(
            x=formatted_causes,
            y=probabilities,
            marker_color=colors,
            text=[f'{p:.1%}' for p in probabilities],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Probability: %{y:.1%}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Root Cause Probability Distribution',
            xaxis_title='Potential Root Cause',
            yaxis_title='Probability',
            yaxis_tickformat='.0%',
            yaxis_range=[0, max(probabilities) * 1.2],
            height=400,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Evidence Used:**")
        for var, value in diagnosis.evidence_used.items():
            st.markdown(f"- {var}: `{value}`")
    
    # Multi-Hypothesis Action Plan
    st.markdown("### 🎯 Recommended Investigation Plan")
    st.markdown("""
    The following actions are ranked by probability. Consider parallel investigation
    for high-probability hypotheses to minimize troubleshooting time.
    """)
    
    for i, action in enumerate(diagnosis.multi_hypothesis_actions, 1):
        # Extract probability from action string
        if '%' in action:
            # Color code by probability
            prob_str = action.split('- ')[-1].split(' probability')[0]
            try:
                prob_val = float(prob_str.strip('%')) / 100
                if prob_val > 0.5:
                    st.error(f"**{i}.** {action}")
                elif prob_val > 0.2:
                    st.warning(f"**{i}.** {action}")
                else:
                    st.info(f"**{i}.** {action}")
            except:
                st.write(f"**{i}.** {action}")
        else:
            st.write(f"**{i}.** {action}")
    
    # Explanation
    with st.expander("📖 Detailed Explanation"):
        st.markdown(diagnosis.explanation)


def render_traditional_diagnosis(diagnosis_results: List[Dict[str, Any]]):
    """Render traditional rule-based diagnosis"""
    st.markdown("### 🔍 Rule-Based Analysis")
    
    for res in diagnosis_results:
        rc = res['root_cause']
        explanation = res['explanation']
        
        with st.expander(
            f"🔴 Root Cause: {rc.root_cause_asset_id} ({int(rc.probability*100)}%)",
            expanded=True
        ):
            st.markdown(explanation)
            st.info(f"**Recommended Action**: {rc.recommended_action}")


def render_granger_causality(causal_graph):
    """Render Granger causality proof section"""
    with st.expander("🔗 Granger Causality Proof (Advanced)", expanded=False):
        st.markdown("""
        **What is Granger Causality?**  
        Statistical hypothesis testing that proves one time-series *causes* another
        (not just correlation). A metric X "Granger-causes" Y if past values of X
        help predict future values of Y beyond what Y's own history provides.
        """)
        
        # Get significant edges
        if hasattr(causal_graph, 'edges') and len(causal_graph.edges) > 0:
            st.markdown("### ✅ Proven Causal Relationships (p < 0.05)")
            
            significant_edges = [
                edge for edge in causal_graph.edges
                if edge.get('p_value', 1.0) < 0.05
            ]
            
            if significant_edges:
                # Create table
                causal_data = []
                for edge in significant_edges:
                    source = edge.get('source', 'Unknown')
                    target = edge.get('target', 'Unknown')
                    p_value = edge.get('p_value', 1.0)
                    lag = edge.get('lag', 0)
                    
                    causal_data.append({
                        'Cause': source,
                        'Effect': target,
                        'p-value': f'{p_value:.4f}',
                        'Lag (time steps)': lag,
                        'Significance': '***' if p_value < 0.01 else '**' if p_value < 0.05 else '*'
                    })
                
                st.table(causal_data)
                
                st.success(f"""
                💡 **Interpretation**: Found {len(significant_edges)} statistically significant
                causal relationships. These are **proven** through hypothesis testing, not assumed
                from network topology.
                """)
                
                # Visualize causal graph
                st.markdown("### 📊 Causal Network Visualization")
                
                # Create network diagram
                fig = go.Figure()
                
                # Get unique nodes
                nodes = set()
                for edge in significant_edges:
                    nodes.add(edge.get('source'))
                    nodes.add(edge.get('target'))
                
                nodes = list(nodes)
                node_positions = {node: i for i, node in enumerate(nodes)}
                
                # Add edges
                for edge in significant_edges:
                    source = edge.get('source')
                    target = edge.get('target')
                    p_value = edge.get('p_value', 1.0)
                    
                    # Line width based on significance
                    width = 3 if p_value < 0.01 else 2
                    
                    fig.add_trace(go.Scatter(
                        x=[node_positions[source], node_positions[target]],
                        y=[0, 0],
                        mode='lines+markers',
                        line=dict(width=width, color='steelblue'),
                        marker=dict(size=10),
                        name=f'{source} → {target}',
                        hovertemplate=f'<b>{source} → {target}</b><br>p={p_value:.4f}<extra></extra>'
                    ))
                
                fig.update_layout(
                    title='Causal Relationships (arrows show direction of causation)',
                    showlegend=False,
                    height=300,
                    xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
                    yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.info("No statistically significant causal relationships detected (p ≥ 0.05)")
        
        else:
            st.warning("""
            ⚠️ Granger causality analysis requires time-series data with ≥30 time points
            per metric. Current dataset may be insufficient.
            
            **To enable**: Generate extended time-series data with:
            ```
            python src/utils/data_generator.py --scenarios 1000 --points 120
            ```
            """)
        
        # Feedback loops
        if hasattr(causal_graph, 'detect_feedback_loops'):
            loops = causal_graph.detect_feedback_loops()
            if loops:
                st.markdown("### 🔄 Detected Feedback Loops")
                st.warning(f"Found {len(loops)} feedback loops in the causal graph:")
                for i, loop in enumerate(loops, 1):
                    st.write(f"{i}. {' → '.join(loop)}")

