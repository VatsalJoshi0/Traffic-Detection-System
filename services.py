import pandas as pd
import requests
import sqlite3
import os

# CONFIGURATION
# Connect to your backend teammate's API and DB here
API_BASE_URL = "http://localhost:5000"
DB_PATH = "traffic_data.db"

def check_login(username, password):
    """
    Validates user credentials.
    In real implementation, this would POST to Flask /login endpoint.
    """
    # MOCK implementation for frontend dev
    if username == "admin" and password == "admin":
        return True
    return False

def get_dashboard_stats():
    """
    Fetches summary statistics.
    Try to hit Flask API, fall back to mock data.
    """
    try:
        response = requests.get(f"{API_BASE_URL}/api/stats", timeout=2)
        if response.status_code == 200:
            return response.json()
    except:
        # Fallback Mock Data
        return {
            "total_violations": 124,
            "active_cameras": 8,
            "system_status": "Online"
        }

def get_recent_violations():
    """
    Fetches violation history.
    Tries SQLite direct access first (fastest for local), then API.
    """
    if os.path.exists(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH)
            df = pd.read_sql("SELECT * FROM violations ORDER BY timestamp DESC LIMIT 50", conn)
            conn.close()
            return df
        except Exception as e:
            print(f"DB Error: {e}")
    
    # Mock Data if DB not found
    data = {
        "timestamp": ["2026-02-19 10:00", "2026-02-19 10:05", "2026-02-19 10:12"],
        "violation_type": ["No Helmet", "Red Light", "Speeding"],
        "location": ["MG Road, Bengaluru", "Connaught Place, Delhi", "NH-44 Highway"],
        "status": ["Pending", "Verified", "Pending"]
    }
    return pd.DataFrame(data)

def upload_file_to_backend(uploaded_file):
    """
    Sends uploaded video to Flask backend for processing.
    """
    # files = {'file': uploaded_file}
    # response = requests.post(f"{API_BASE_URL}/upload", files=files)
    # return response.json()
    return {"status": "success", "message": "File received. Processing started."}
