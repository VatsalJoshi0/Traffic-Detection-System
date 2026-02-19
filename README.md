# 🚦 AI-Based Traffic Detection System

> **Code Unnati Innovation Program**  
> *Automating Road Safety with Computer Vision & Deep Learning*

![Traffic Sentinel Banner](https://images.unsplash.com/photo-1566835848608-89c030d97274?q=80&w=1000&auto=format&fit=crop)

## 📖 Project Overview
Road traffic violations are a major cause of accidents globally. This project, **Traffic Sentinel AI**, leverages **Deep Learning (YOLO)** and **Computer Vision** to automatically detect violations from video feeds.

This repository contains the **Frontend Dashboard** built with **Streamlit**, designed to provide Traffic Police with real-time alerts and analytics.

### 🌟 Key Features
- **Real-time Monitoring:** Live video feed analysis with bounding boxes.
- **Violation Detection:** Automatically flags:
  - ⛑️ No Helmet
  - 🚦 Red Light Jump
  - 🏎️ Over-speeding
- **Indian Context:** Tailored for Indian roads (e.g., MG Road, Connaught Place locations).
- **Interactive Dashboard:** Filter history, view stats, and export reports.

---

## 🛠️ Tech Stack
| Component | Technology |
| :--- | :--- |
| **Frontend** | Streamlit (Python) |
| **Backend** | Flask (API) |
| **Database** | SQLite |
| **AI Model** | YOLOv8 + OpenCV |

---

## 🚀 How to Run the Frontend
Follow these steps to set up the dashboard on your local machine.

### 1. Prerequisites
- Python 3.8 or higher installed.

### 2. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/YOUR_USERNAME/Traffic-Detection-System.git
cd Traffic-Detection-System
pip install -r requirements.txt
```

### 3. Launch the App
Run the following command to start the Streamlit server:
```bash
streamlit run app.py
```
*Alternatively, on Windows, just double-click `run_app.bat`.*

### 4. Access the Dashboard
Open your browser and navigate to:
`http://localhost:8501`

**Default Credentials:**
- **Username:** `admin`
- **Password:** `admin`

---

## 📂 Project Structure
```
traffic_app/
├── app.py                # Main Application Entry Point
├── services.py           # API Integration Layer
├── requirements.txt      # Python Dependencies
├── run_app.bat           # Quick Launcher
└── modules/              # UI Components
    ├── login.py          # Authentication
    ├── dashboard.py      # Live Monitoring
    ├── file_upload.py    # Video Analysis
    └── support.py        # Help Center
```

## 👥 Contributors
- **[Your Name]** - Frontend Developer (Streamlit)
- **[Teammate Name]** - Backend Developer (Flask/AI)

---

## 📜 License
This project is part of the Code Unnati Innovation Program.
