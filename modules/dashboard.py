import streamlit as st
from services import get_dashboard_stats, get_recent_violations
import pandas as pd

def render_dashboard():
    st.header("🖥️ Realtime Monitoring Dashboard")
    
    # Top Stats Row
    stats = get_dashboard_stats()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Violations (Today)", stats['total_violations'], "+12%")
    c2.metric("Active Cameras", stats['active_cameras'], "Online")
    c3.metric("System Status", stats['system_status'])
    
    st.divider()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📍 Live Camera Feed")
        # Placeholder for Flask Stream
        # In real integration: st.image("http://localhost:5000/video_feed")
        st.image("https://images.unsplash.com/photo-1566835848608-89c030d97274?q=80&w=1000&auto=format&fit=crop", caption="Camera 01 - MG Road Junction (Live)", use_container_width=True)
        
    with col2:
        st.subheader("🔔 Recent Alerts")
        df = get_recent_violations()
        
        # Simple list view for alerts
        for i, row in df.iterrows():
            with st.container():
                st.warning(f"**{row['violation_type']}** at {row['location']}")
                st.caption(f"Time: {row['timestamp']} | Status: {row['status']}")
                
    st.divider()
    
    st.subheader("📊 Violation Trends")
    st.bar_chart(df['violation_type'].value_counts())
