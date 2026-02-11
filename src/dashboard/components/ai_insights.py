import streamlit as st
from typing import List, Dict, Any

def render_ai_insights(diagnosis_results: List[Dict[str, Any]]):
    st.subheader("🤖 AI Root Cause Analysis")
    
    if not diagnosis_results:
        st.info("No active root causes identified. System looks healthy!")
        return
        
    for res in diagnosis_results:
        rc = res['root_cause']
        explanation = res['explanation']
        
        with st.expander(f"🔴 Root Cause: {rc.root_cause_asset_id} ({int(rc.probability*100)}%)", expanded=True):
            st.markdown(explanation)
            st.info(f"**Recommended Action**: {rc.recommended_action}")
