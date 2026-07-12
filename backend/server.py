import os
import cv2
import time
import logging
import threading
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

import sys
from pathlib import Path
# Ensure we can import from the same directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vision_pipeline import VisionPipeline
from database import init_db, _get_connection, db_lock

# ---------------------------------------------------------------------------
# Logging & Initialization
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("SentinelVision.Server")

# Initialize the database and ensure tables exist
init_db()

# Global pipeline instance
pipeline = VisionPipeline(
    model_path="models/yolov8n.pt",
    flow_direction="top_to_bottom",
    confidence_threshold=0.45
)

# ---------------------------------------------------------------------------
# Thread Isolation & Shared State
# ---------------------------------------------------------------------------
# The background processing loop is now encapsulated within VisionPipeline.
# The `/video_feed` generator will read directly from the pipeline's thread-safe buffer.


# ---------------------------------------------------------------------------
# FastAPI Application & Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages the startup and shutdown of the background processing thread."""
    pipeline.start_ingestion_loop()
    yield
    pipeline.stop_ingestion_loop()

app = FastAPI(title="SentinelVision High-Concurrency API", lifespan=lifespan)

# ---------------------------------------------------------------------------
# Async Endpoints
# ---------------------------------------------------------------------------
def frame_generator():
    """
    Global streaming generator yielding multipart JPEG frames.
    Reads from the pipeline's thread-isolated `latest_jpeg` buffer.
    """
    while True:
        with pipeline.jpeg_lock:
            frame = pipeline.latest_jpeg
            
        if not frame:
            time.sleep(0.05)
            continue
            
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
        )
        # Yield to event loop, maintaining ~30fps stream
        time.sleep(1.0 / 30.0)


@app.get("/video_feed")
async def video_feed():
    """
    Emits the processed OpenCV image matrices from vision_pipeline.py 
    as a fast MJPEG stream using the boundary multipart protocol.
    """
    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    """
    Receives a video file upload, saves it locally, and redirects the 
    VisionPipeline to ingest from this new file dynamically.
    """
    os.makedirs("assets/sample_videos", exist_ok=True)
    file_path = f"assets/sample_videos/{file.filename}"
    
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
        
    pipeline.set_video_source(file_path)
    return {"status": "success", "source": file_path}


@app.get("/reset_to_live_cam")
async def reset_to_live_cam():
    """
    Resets the active video source of the VisionPipeline back to the 
    default live webcam (index 0).
    """
    pipeline.set_video_source(0)
    return {"status": "success", "source": 0}


class SignalStatusUpdate(BaseModel):
    flow_direction: str


@app.post("/set_signal_status")
async def set_signal_status(status: SignalStatusUpdate):
    """
    Updates intended direction sliders / flow parameters asynchronously.
    """
    pipeline.flow_direction = status.flow_direction
    return {"status": "success", "flow_direction": pipeline.flow_direction}


class OverrideTrigger(BaseModel):
    active: bool


@app.post("/trigger_override")
async def trigger_override(trigger: OverrideTrigger):
    """
    Triggers or clears the manual emergency override natively in the CV pipeline.
    """
    pipeline.trigger_manual_override(trigger.active)
    return {"status": "success", "emergency_active": pipeline.emergency_active}


@app.get("/get_metrics")
async def get_metrics() -> Dict[str, Any]:
    """
    Retrieves current traffic light status and system metrics without 
    interrupting frame-processing workflows.
    """
    metrics = {
        "emergency_override_active": pipeline.emergency_active,
        "current_flow_direction": pipeline.flow_direction,
        "signal_data": None,
        "total_violations": 0
    }
    
    with db_lock:
        conn = _get_connection()
        try:
            cursor = conn.cursor()
            
            # Fetch latest traffic signal density metric
            cursor.execute("SELECT * FROM traffic_signals ORDER BY id DESC LIMIT 1")
            signal_row = cursor.fetchone()
            if signal_row:
                metrics["signal_data"] = dict(signal_row)
                
            # Fetch aggregate violations count
            cursor.execute("SELECT COUNT(*) as cnt FROM violations")
            v_row = cursor.fetchone()
            if v_row:
                metrics["total_violations"] = v_row["cnt"]
                
        except Exception as e:
            logger.error(f"Error executing DB metric query: {e}")
        finally:
            conn.close()
            
    return metrics


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
