import os
import sys
import time
import requests
import pandas as pd
import streamlit as st

# Ensure we can import from the backend directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend")))

from backend.database import get_violations_by_plate

# ---------------------------------------------------------------------------
# Streamlit Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SentinelVision - Interactive Traffic Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BACKEND_URL = "http://localhost:8000"

# ---------------------------------------------------------------------------
# Custom CSS for Premium Dark UI
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Styling headers & titles */
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00FFCC 0%, #0077FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    /* Metrics cards styling */
    .metric-card {
        background: rgba(255, 255, 255, 0.04);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.4);
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: rgba(0, 255, 204, 0.4);
    }
    .metric-title {
        color: #aaaaaa;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
        margin-top: 8px;
    }
    .metric-desc {
        font-size: 0.75rem;
        color: #00FFCC;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# API Helper Methods
# ---------------------------------------------------------------------------
def fetch_metrics():
    """Fetches real-time telemetry metrics from the FastAPI backend."""
    try:
        response = requests.get(f"{BACKEND_URL}/get_metrics", timeout=1.5)
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.RequestException:
        pass
    return None


# ---------------------------------------------------------------------------
# Sidebar UI & Controls
# ---------------------------------------------------------------------------
st.sidebar.markdown("## 🛡️ Control Parameters")
st.sidebar.write("Configure edge variables natively transmitted to the vision pipeline.")

# 1. Fetch current backend states for synchronization
backend_data = fetch_metrics()
current_direction = backend_data.get("current_flow_direction", "top_to_bottom") if backend_data else "top_to_bottom"
current_override = backend_data.get("emergency_override_active", False) if backend_data else False

# 2. Flow direction dropdown
direction_options = ["top_to_bottom", "bottom_to_top"]
flow_dir = st.sidebar.selectbox(
    "Intended Flow Direction",
    direction_options,
    index=direction_options.index(current_direction)
)

if flow_dir != current_direction:
    try:
        res = requests.post(f"{BACKEND_URL}/set_signal_status", json={"flow_direction": flow_dir}, timeout=2.0)
        if res.status_code == 200:
            st.sidebar.success(f"Flow updated to {flow_dir}")
            st.rerun()
    except requests.exceptions.RequestException:
        st.sidebar.error("Failed to connect to backend control endpoint.")

# 3. Force Emergency Override toggle
emergency_toggle = st.sidebar.checkbox(
    "Force Emergency Override Lock",
    value=current_override,
    help="Locks system state and disables standard violation tracking."
)

if emergency_toggle != current_override:
    try:
        res = requests.post(f"{BACKEND_URL}/trigger_override", json={"active": emergency_toggle}, timeout=2.0)
        if res.status_code == 200:
            st.sidebar.success(f"Override toggled: {emergency_toggle}")
            st.rerun()
    except requests.exceptions.RequestException:
        st.sidebar.error("Failed to transmit emergency command to backend.")

# 4. Video Source Management
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎥 Video Source Management")

uploaded_video = st.sidebar.file_uploader("Upload Traffic Video (.mp4)", type=["mp4", "mov", "avi"])

if uploaded_video is not None:
    if st.session_state.get("last_uploaded") != uploaded_video.name:
        with st.spinner("Uploading and processing video..."):
            files = {"file": (uploaded_video.name, uploaded_video.getvalue(), uploaded_video.type)}
            try:
                res = requests.post(f"{BACKEND_URL}/upload_video", files=files, timeout=10.0)
                if res.status_code == 200:
                    st.session_state["last_uploaded"] = uploaded_video.name
                    st.sidebar.success("Video uploaded and pipeline updated!")
                    time.sleep(1) # Give user time to see success message
                    st.rerun()
                else:
                    st.sidebar.error("Failed to upload video.")
            except requests.exceptions.RequestException:
                st.sidebar.error("Could not reach backend for upload.")

if st.sidebar.button("Switch to Live Camera Feed", use_container_width=True):
    try:
        res = requests.get(f"{BACKEND_URL}/reset_to_live_cam", timeout=2.0)
        if res.status_code == 200:
            st.session_state["last_uploaded"] = None
            st.sidebar.success("Switched to Live Camera (index 0)")
            time.sleep(1)
            st.rerun()
    except requests.exceptions.RequestException:
        st.sidebar.error("Could not reach backend to reset camera.")


# ---------------------------------------------------------------------------
# Main Tabs Layout
# ---------------------------------------------------------------------------
tab1, tab2 = st.tabs(["📊 Live Intersection Dashboard", "🚗 Citizen Portal & VAHAN Gateway"])

# ───────────────────────────────────────────────────────────────────────────
# TAB 1: Live Dashboard
# ───────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("<h1 class='main-title'>SentinelVision Edge Dashboard</h1>", unsafe_allow_html=True)
    st.write("Real-time edge telemetry and video ingestion processing.")
    
    col_feed, col_telemetry = st.columns([13, 7])
    
    with col_feed:
        st.markdown("### 🎥 Camera Pipeline stream")
        # Video feed pulling from the FastAPI server (No native YOLO/OpenCV tracking here)
        st.image("http://localhost:8000/video_feed", caption="Live Processed Feed (640×480 Boundary Space)", use_container_width=True)
        
    with col_telemetry:
        st.markdown("### 📈 Live Intersection Telemetry")
        
        # Dashboard manual refresh button
        if st.button("Refresh Telemetry Indicators"):
            st.rerun()
            
        metrics = fetch_metrics()
        
        if metrics:
            signal_data = metrics.get("signal_data")
            density = signal_data.get("current_density", 0.0) if signal_data else 0.0
            green_time = signal_data.get("allocated_green_time", 30) if signal_data else 30
            total_violations = metrics.get("total_violations", 0)
            override_state = "ACTIVE" if metrics.get("emergency_override_active") else "INACTIVE"
            override_color = "#FF3333" if override_state == "ACTIVE" else "#00FFCC"
            
            st.markdown(f"""
            <div style='display: grid; grid-template-columns: 1fr; gap: 15px; margin-top: 10px;'>
                <div class='metric-card'>
                    <div class='metric-title'>Active Violations</div>
                    <div class='metric-value'>{total_violations}</div>
                    <div class='metric-desc'>Logged in database</div>
                </div>
                <div class='metric-card'>
                    <div class='metric-title'>Current Vehicle Density</div>
                    <div class='metric-value'>{int(density)}</div>
                    <div class='metric-desc'>Edge-counted in camera view</div>
                </div>
                <div class='metric-card'>
                    <div class='metric-title'>Allocated Green Time</div>
                    <div class='metric-value'>{green_time}s</div>
                    <div class='metric-desc'>Calculated adaptively</div>
                </div>
                <div class='metric-card'>
                    <div class='metric-title'>System Override State</div>
                    <div class='metric-value' style='color: {override_color};'>{override_state}</div>
                    <div class='metric-desc'>Locked during emergency events</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ FastAPI Backend Telemetry Offline. Start `backend/server.py` to activate dashboards.")

# ───────────────────────────────────────────────────────────────────────────
# TAB 2: Citizen Portal & VAHAN Gateway
# ───────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("<h2 class='main-title'>🚗 Citizen Challan & VAHAN Gateway</h2>", unsafe_allow_html=True)
    st.write("Search infraction records by vehicle plate and authenticate to view evidence crops.")
    
    plate_input = st.text_input("Enter Vehicle Registration ID (e.g. TRACK_2):")
    
    if plate_input:
        cleaned_plate = plate_input.strip()
        
        # Read from database hook
        try:
            violations = get_violations_by_plate(cleaned_plate)
        except Exception as e:
            st.error(f"Failed to communicate with DB: {e}")
            violations = []
            
        if violations:
            st.success(f"Found {len(violations)} infraction record(s) matching '{cleaned_plate}'")
            
            st.markdown("### 📋 Masked VAHAN Registration Details")
            st.markdown(f"""
            <div class='metric-card' style='margin-bottom: 20px;'>
                <table style='width: 100%; border-collapse: collapse;'>
                    <tr>
                        <td style='padding: 6px 0; color: #888888; width: 40%;'>Registered Owner Name:</td>
                        <td style='padding: 6px 0; font-weight: bold;'>A****** K**** S****</td>
                    </tr>
                    <tr>
                        <td style='padding: 6px 0; color: #888888;'>Registered Mobile:</td>
                        <td style='padding: 6px 0; font-weight: bold;'>+91 ******9988</td>
                    </tr>
                    <tr>
                        <td style='padding: 6px 0; color: #888888;'>Registration RTO:</td>
                        <td style='padding: 6px 0; font-weight: bold;'>MH-12-**-**** (Pune Central)</td>
                    </tr>
                    <tr>
                        <td style='padding: 6px 0; color: #888888;'>Chassis Number / VIN:</td>
                        <td style='padding: 6px 0; font-weight: bold;'>MA3D***************</td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
            
            # Initialize Session State Variables for OTP
            if "otp_sent" not in st.session_state:
                st.session_state.otp_sent = False
            if "otp_verified" not in st.session_state:
                st.session_state.otp_verified = False
                
            if not st.session_state.otp_verified:
                st.info("Verification is required to unlock exact time-series infraction records and cropped evidence images.")
                if st.button("Send VAHAN OTP"):
                    st.session_state.otp_sent = True
                    st.info("OTP successfully sent to registered mobile number +91 ******9988")
                    
                if st.session_state.otp_sent:
                    otp_code = st.text_input("Enter 4-Digit VAHAN OTP (Demo Code: 1234):")
                    if st.button("Verify Credentials"):
                        if otp_code == "1234":
                            st.session_state.otp_verified = True
                            st.success("Verification successful! Credentials unlocked.")
                            st.rerun()
                        else:
                            st.error("Invalid OTP code. Please try again.")
            else:
                # Credentials verified -> show violations database records & crops
                st.markdown("### 🚨 Verified Infraction History")
                
                df_violations = pd.DataFrame(violations)
                # Select display columns
                cols_to_show = ["id", "timestamp", "violation_type", "confidence", "status"]
                filtered_cols = [c for c in cols_to_show if c in df_violations.columns]
                st.dataframe(df_violations[filtered_cols], use_container_width=True)
                
                st.markdown("### 📸 Evidence Crops & Chronological Data")
                for _, row in df_violations.iterrows():
                    col_crop, col_details = st.columns([1, 2])
                    
                    with col_crop:
                        img_path = row.get("image_path")
                        if img_path and os.path.exists(img_path):
                            st.image(img_path, caption=f"Evidence Crop (ID: {row.get('id')})", use_container_width=True)
                        else:
                            st.warning("Evidence file not found on local edge disk.")
                            
                    with col_details:
                        st.write(f"**Violation ID:** `{row.get('id')}`")
                        st.write(f"**Infraction Category:** {row.get('violation_type')}")
                        st.write(f"**Timestamp:** {row.get('timestamp')}")
                        st.write(f"**Confidence Level:** {row.get('confidence'):.2f}")
                        st.write(f"**Status:** `{row.get('status')}`")
                    st.divider()
                    
                if st.button("Clear VAHAN Session"):
                    st.session_state.otp_sent = False
                    st.session_state.otp_verified = False
                    st.rerun()
        else:
            st.warning(f"No infraction records logged for registration plate: '{cleaned_plate}'")
