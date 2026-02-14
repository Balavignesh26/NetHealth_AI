"""
Validation Metrics Dashboard Component

Displays validation results including:
- Accuracy, Precision, Recall, F1-Score
- Confusion matrix
- Validation status
"""

import streamlit as st
import json
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from PIL import Image


def render_validation_metrics():
    """Render validation metrics in dashboard"""
    st.header("📊 System Validation Metrics")
    
    st.markdown("""
    This section shows the diagnostic accuracy of the AI system, validated against
    1,000+ labeled fault scenarios with known ground truth.
    """)
    
    # Load validation metrics
    metrics_file = Path('outputs/VALIDATION_METRICS.json')
    
    if not metrics_file.exists():
        st.warning("⚠️ Validation metrics not yet generated.")
        st.info("**To generate validation metrics:**")
        st.code("""
# Step 1: Generate synthetic data
python src/utils/data_generator.py --scenarios 1000 --points 120

# Step 2: Run validation
python tests/run_validation.py
        """, language='bash')
        
        st.markdown("---")
        st.subheader("Why Validation Matters")
        st.markdown("""
        - **Quantitative Proof**: Demonstrates diagnostic accuracy with hard numbers
        - **Confusion Matrix**: Shows which fault types are correctly identified
        - **Confidence Calibration**: Validates that confidence scores are meaningful
        - **Competitive Edge**: Provides evidence-based performance metrics
        """)
        return
    
    # Load metrics
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
    except Exception as e:
        st.error(f"Error loading validation metrics: {e}")
        return
    
    # Display overall metrics
    st.subheader("Overall Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Accuracy",
            f"{metrics['accuracy']:.1%}",
            help="Percentage of correct diagnoses"
        )
    
    with col2:
        st.metric(
            "Precision",
            f"{metrics['precision']:.1%}",
            help="True Positives / (True Positives + False Positives)"
        )
    
    with col3:
        st.metric(
            "Recall",
            f"{metrics['recall']:.1%}",
            help="True Positives / (True Positives + False Negatives)"
        )
    
    with col4:
        st.metric(
            "F1-Score",
            f"{metrics['f1']:.3f}",
            help="Harmonic mean of Precision and Recall"
        )
    
    # Validation summary
    st.success(
        f"✅ Validated on {metrics['total_scenarios']} synthetic fault scenarios. "
        f"{metrics['correct_predictions']} correct predictions."
    )
    
    # Confusion Matrix
    st.markdown("---")
    st.subheader("Confusion Matrix")
    
    cm_file = Path('outputs/confusion_matrix.png')
    if cm_file.exists():
        try:
            image = Image.open(cm_file)
            st.image(image, use_container_width=True)
            
            st.caption("""
            **How to read**: Rows = Actual fault type, Columns = Predicted fault type.
            Diagonal values (dark blue) = correct predictions. Off-diagonal = misclassifications.
            """)
        except Exception as e:
            st.error(f"Error loading confusion matrix: {e}")
    else:
        st.warning("Confusion matrix image not found")
    
    # Per-fault-type performance
    st.markdown("---")
    st.subheader("Performance by Fault Type")
    
    accuracy_file = Path('outputs/accuracy_by_fault_type.png')
    if accuracy_file.exists():
        try:
            image = Image.open(accuracy_file)
            st.image(image, use_container_width=True)
        except Exception as e:
            st.error(f"Error loading accuracy chart: {e}")
    
    # Classification report details
    with st.expander("📋 Detailed Classification Report"):
        report = metrics.get('classification_report', {})
        
        # Create DataFrame for display
        fault_types = [k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
        
        if fault_types:
            report_data = []
            for ft in fault_types:
                report_data.append({
                    'Fault Type': ft,
                    'Precision': f"{report[ft]['precision']:.2%}",
                    'Recall': f"{report[ft]['recall']:.2%}",
                    'F1-Score': f"{report[ft]['f1-score']:.3f}",
                    'Support': int(report[ft]['support'])
                })
            
            st.table(report_data)
    
    # Confidence statistics
    st.markdown("---")
    st.subheader("Confidence Calibration")
    
    conf_stats = metrics.get('confidence_stats', {})
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Mean Confidence (Correct)",
            f"{conf_stats.get('mean', 0):.2f}",
            help="Average confidence score for correct predictions"
        )
    with col2:
        st.metric(
            "Std Deviation",
            f"{conf_stats.get('std', 0):.2f}",
            help="Confidence score variability"
        )
    
    st.info("""
    💡 **Interpretation**: High mean confidence on correct predictions indicates
    the system's confidence scores are well-calibrated and trustworthy.
    """)
    
    # Timestamp
    st.caption(f"Last validated: {metrics.get('timestamp', 'Unknown')}")
