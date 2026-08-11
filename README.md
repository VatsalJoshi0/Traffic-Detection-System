# SentinelVision AI 🚦👁️

> **Next-Generation Computer Vision & AI Traffic Monitoring System**

SentinelVision AI is an intelligent, real-time traffic detection and violation analysis platform powered by YOLO computer vision models, SQLite database architecture, and a modern Streamlit interactive dashboard.

---

## 🌟 Key Features

- **Real-Time Multi-Violation Detection**:
  - 🪖 **Helmet Violation**: Identifies two-wheeler riders without helmets.
  - 🚗 **Seatbelt Compliance**: Tracks drivers/passengers for seatbelt compliance.
  - 🚦 **Red Light / Wrong Lane Infringement**: Detects lane discipline and traffic signal violations.
  - 📱 **Mobile Phone Usage**: Flags drivers using mobile devices while operating vehicles.

- **Dynamic Adaptive Traffic Signal Control**:
  - Calculates real-time vehicle density at intersections.
  - Dynamically optimizes and allocates green light durations to reduce traffic congestion.

- **Interactive Operations Dashboard**:
  - Live video stream analytics with dynamic visual bounding box overlays.
  - Automated evidence cropping and instant violation logging.
  - Filterable violation history logs with downloadable reports and database search.
  - Intersection performance metrics and signal status visualizers.

---

## 🏗️ System Architecture

SentinelVision AI is built with a clean, decoupled modular architecture:

```
SentinelVision_Production/
├── backend/
│   ├── vision_pipeline.py  # Computer vision inference & detection pipeline
│   ├── database.py         # SQLite database management & traffic logs
│   └── server.py           # REST API & background processing service
├── frontend/
│   └── app.py              # Streamlit Web UI dashboard
├── models/                 # Pre-trained YOLO models & weights
├── assets/                 # Evidence crop snapshots & sample video files
└── sentinel_vision.db      # Local SQLite database
```

---

## ⚡ Quick Start & Installation

### 1. Prerequisites
- **Python 3.9+** installed
- **Git**
- **NVIDIA GPU + CUDA** (Optional, for higher FPS detection)

### 2. Clone the Repository
```bash
git clone https://github.com/VatsalJoshi0/Traffic-Detection-System.git
cd Traffic-Detection-System
```

### 3. Create & Activate Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
# Or install core packages:
pip install ultralytics opencv-python streamlit pandas pillow
```

---

## 🚀 Running the Application

### Launch the Streamlit Frontend Dashboard:
```bash
streamlit run frontend/app.py
```
Open your browser and navigate to `http://localhost:8501`.

### (Optional) Launch Backend Service / Database Engine:
```bash
python backend/server.py
```

---

## 📄 License & Acknowledgments

Distributed under the **MIT License**. See `LICENSE` for more information.

Developed for smart city traffic management and intelligent transportation monitoring.
