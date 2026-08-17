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
│   └── server.py           # FastAPI REST API & background processing service
├── frontend/
│   └── app.py              # Streamlit Web UI dashboard
├── models/                 # Pre-trained YOLO models & weights (auto-downloaded)
├── assets/                 # Evidence crop snapshots & sample video files
│   ├── evidence_crops/     # Real-time violation snapshot crops (auto-generated)
│   └── sample_videos/      # Demo & benchmark video streams
├── sentinel_vision.db      # Local SQLite database (auto-generated at runtime)
├── requirements.txt        # Production Python dependencies
├── EVALUATION_AND_TEST_REPORT.md # Comprehensive test & benchmark report
└── LICENSE                 # MIT License
```

---

## ⚡ Quick Start & Installation

### 1. Prerequisites
- **Python 3.9+** installed
- **Git**
- **NVIDIA GPU + CUDA** (Optional, for accelerated GPU inference)

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
```

> **Note on Model Weights**: Standard YOLOv8 weights (`yolov8n.pt`) are downloaded automatically by the pipeline on first launch. You can also place custom fine-tuned weights directly in the `models/` directory.

---

## 🚀 Running the Application

### 1. Launch the Streamlit Frontend Dashboard:
```bash
streamlit run frontend/app.py
```
Open your browser and navigate to `http://localhost:8501`.

### 2. (Optional) Launch the FastAPI Backend Service:
```bash
python backend/server.py
```
API docs and endpoints will be available at `http://localhost:8000/docs`.

---

## 📊 Evaluation & Benchmarks

For in-depth performance benchmarks, mean Average Precision (mAP), inference latency, and violation detection test results, see the [Evaluation and Test Report](EVALUATION_AND_TEST_REPORT.md).

---

## 📄 License & Acknowledgments

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more information.

Developed for smart city traffic management and intelligent transportation monitoring.
