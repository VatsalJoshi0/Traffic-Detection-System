"""
SentinelVision — backend/vision_pipeline.py
Principal Computer Vision Engineer Implementation

Responsibilities:
  - Wrong-lane vector analysis via rolling trajectory slope
  - Triple-riding & head/helmet isolation engine
  - Seatbelt / phone obstruction anomaly detection
  - Emergency vehicle lock-override and deferred state cleanup
  - Adaptive signal metric updates via density measurement
  - Thread-safe event logging through backend/database.py

Processing canvas: all inference is performed on frames resized to 640×480.
ROI polygon coordinates are declared relative to this canvas.
"""

from __future__ import annotations

import os
import cv2
import time
import logging
import threading
import numpy as np
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from ultralytics import YOLO

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from database import (
    insert_violation,
    insert_accident,
    update_signal_metrics,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("SentinelVision.VisionPipeline")

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------

# ── Processing canvas ──────────────────────────────────────────────────────
PROCESS_W: int = 640
PROCESS_H: int = 480

# ── ROI polygon (coords relative to 640×480 canvas) ───────────────────────
# Represents a virtual intersection / lane-boundary polygon.
# Adjust vertices to match your physical camera calibration.
ROI_POLYGON: np.ndarray = np.array([
    [100, 60],   # top-left
    [540, 60],   # top-right
    [590, 420],  # bottom-right
    [50,  420],  # bottom-left
], dtype=np.int32)

# ── YOLO class indices (COCO-80 defaults) ──────────────────────────────────
CLASS_PERSON:      int = 0
CLASS_BICYCLE:     int = 1
CLASS_CAR:         int = 2
CLASS_MOTORCYCLE:  int = 3
CLASS_BUS:         int = 5
CLASS_TRUCK:       int = 7
CLASS_AMBULANCE:   int = 7   # override / retrain for custom ambulance class

# Vehicle classes eligible for in-cabin checks
VEHICLE_CLASSES: Tuple[int, ...] = (CLASS_CAR, CLASS_BUS, CLASS_TRUCK)

# ── Wrong-lane analysis parameters ─────────────────────────────────────────
TRAJECTORY_WINDOW:    int   = 30    # rolling frames for slope computation
WRONG_LANE_DELTA_Y:   int   = -30  # px: upward displacement threshold (top-to-bottom flow)
WRONG_LANE_COOLDOWN:  int   = 60   # frames before re-triggering for same track ID

# ── Triple-riding parameters ────────────────────────────────────────────────
MAX_RIDERS:           int   = 2     # more than this on one motorcycle → violation
HEAD_ZONE_FRACTION:   float = 0.20  # upper 20 % of person-box = head/helmet zone

# ── Seatbelt / phone anomaly parameters ────────────────────────────────────
CABIN_TOP_FRAC:    float = 0.15   # top 15 % of vehicle box = cabin top edge skip
CABIN_BOTTOM_FRAC: float = 0.50   # bottom boundary of upper-mid cabin slice
CABIN_LEFT_FRAC:   float = 0.25   # horizontal margins to isolate cabin interior
CABIN_RIGHT_FRAC:  float = 0.75
CABIN_DARK_THRESH: float = 55.0   # mean pixel brightness below this → obstruction
CABIN_CONF_THRESH: float = 0.40   # minimum YOLO box confidence for vehicle

# ── Emergency override parameters ──────────────────────────────────────────
EMERGENCY_DROPOUT_RESET: int = 30  # frames after ambulance leaves → full state reset

# ── Asset directories ──────────────────────────────────────────────────────
EVIDENCE_DIR: Path = Path("assets/evidence_crops")
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

# ── Mock GPS coordinate for this camera node ───────────────────────────────
CAMERA_LAT_LONG_MOCK: str = "28.6139,77.2090"

# ── Adaptive signal parameters ─────────────────────────────────────────────
JUNCTION_NAME:        str = os.environ.get("JUNCTION_NAME", "Junction_Alpha")
BASE_GREEN_TIME:      int = 30    # seconds
DENSITY_PER_SECOND:   int = 2     # extra green seconds per detected vehicle

# ---------------------------------------------------------------------------
# TrackState — per-ID trajectory and cooldown bookkeeping
# ---------------------------------------------------------------------------

class TrackState:
    """Stores rolling coordinate history and violation cooldown for one tracking ID."""

    def __init__(self, track_id: int, window: int = TRAJECTORY_WINDOW) -> None:
        self.track_id: int = track_id
        self.coords: deque = deque(maxlen=window)  # (cx, cy) tuples
        self.wrong_lane_cooldown: int = 0           # frames remaining before re-trigger
        self.last_seen_frame: int = 0

    def push(self, cx: float, cy: float) -> None:
        self.coords.append((cx, cy))

    def trajectory_delta_y(self) -> Optional[float]:
        """
        Returns net vertical displacement over the full rolling window.
        Positive → downward movement; Negative → upward movement.
        Returns None when the window is not yet full.
        """
        if len(self.coords) < self.coords.maxlen:
            return None
        first_cy = self.coords[0][1]
        last_cy  = self.coords[-1][1]
        return last_cy - first_cy

    def tick_cooldown(self) -> None:
        if self.wrong_lane_cooldown > 0:
            self.wrong_lane_cooldown -= 1


# ---------------------------------------------------------------------------
# VisionPipeline — core orchestrator
# ---------------------------------------------------------------------------

class VisionPipeline:
    """
    Processes a video stream or file frame-by-frame, detects traffic violations,
    logs events to the database, and emits annotated frames.

    Parameters
    ----------
    model_path : str
        Path to the YOLOv8/v9/v11 `.pt` weights file.
    flow_direction : str
        Expected dominant traffic flow direction.
        'top_to_bottom' — vehicles move from top toward bottom of frame.
        'bottom_to_top' — vehicles move from bottom toward top of frame.
    confidence_threshold : float
        Minimum detection confidence to accept a box (0–1).
    manual_override : bool
        When True, acts as if an emergency signal is perpetually active.
    """

    def __init__(
        self,
        model_path: str = "models/yolov8n.pt",
        flow_direction: str = "top_to_bottom",
        confidence_threshold: float = 0.45,
        manual_override: bool = False,
    ) -> None:
        logger.info("Initialising VisionPipeline (model=%s, flow=%s)", model_path, flow_direction)

        self.model: YOLO = YOLO(model_path)
        self.flow_direction: str = flow_direction
        self.conf_thresh: float = confidence_threshold

        # ── Per-track state registry ─────────────────────────────────────
        self._track_states: Dict[int, TrackState] = {}
        self._state_lock = threading.Lock()

        # ── Emergency override state ─────────────────────────────────────
        self.emergency_active: bool = manual_override
        self._emergency_track_ids: set = set()
        self._emergency_dropout_counters: Dict[int, int] = {}  # track_id → frames left

        # ── Frame counter (monotonically increasing) ─────────────────────
        self._frame_index: int = 0

        # ── Violation deduplication: set of (track_id, violation_type) ──
        self._logged_events: set = set()
        
        # ── Ingestion Loop State ─────────────────────────────────────────
        self.video_source: Any = 0
        self.latest_jpeg: bytes = b''
        self.jpeg_lock = threading.Lock()
        self.is_running: bool = False
        self._ingestion_thread: Optional[threading.Thread] = None

        logger.info("VisionPipeline ready. ROI polygon:\n%s", ROI_POLYGON)

    def reset_state(self) -> None:
        """
        Clears tracking histories, logs, and metrics to prevent cross-contamination
        when switching video streams or restarting video files.
        """
        with self._state_lock:
            self._track_states.clear()
            self._logged_events.clear()
            self._emergency_track_ids.clear()
            self._emergency_dropout_counters.clear()
            self._frame_index = 0
            # If emergency override was active due to an ambulance (not manual), clear it
            if not getattr(self, '_manual_override_explicit', False):
                self.emergency_active = False
        logger.info("Pipeline state cleanly flushed.")

    def set_video_source(self, source: Any) -> None:
        """
        Dynamically updates the active video source. 
        Will take effect on the next frame read in the ingestion worker.
        """
        logger.info("Setting video source to: %s", source)
        self.video_source = source
        self.reset_state()

    def start_ingestion_loop(self) -> None:
        """Starts the dedicated background CV ingestion loop thread."""
        if not self.is_running:
            self.is_running = True
            self._ingestion_thread = threading.Thread(target=self._ingestion_worker, daemon=True)
            self._ingestion_thread.start()

    def stop_ingestion_loop(self) -> None:
        """Stops the ingestion loop gracefully."""
        self.is_running = False
        if self._ingestion_thread:
            self._ingestion_thread.join(timeout=3.0)

    def _ingestion_worker(self) -> None:
        """
        Dedicated background worker that continuously reads from self.video_source,
        processes frames, and caches MJPEG bytes into self.latest_jpeg.
        """
        current_source = None
        cap = None

        while self.is_running:
            if self.video_source != current_source:
                if cap is not None:
                    cap.release()
                current_source = self.video_source
                cap = cv2.VideoCapture(current_source)
                if not cap.isOpened():
                    logger.error("Failed to open source: %s. Falling back to webcam.", current_source)
                    current_source = 0
                    self.video_source = 0
                    cap = cv2.VideoCapture(0)

            start_time = time.time()
            ret, frame = cap.read()

            if not ret:
                logger.info("Video stream ended. Resetting state.")
                self.reset_state()
                if isinstance(current_source, str):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop file
                    continue
                else:
                    time.sleep(1) # Wait before retrying camera
                    continue

            # Process frame
            annotated_frame = self.process_frame(frame)
            
            # Encode for MJPEG streaming
            success, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if success:
                with self.jpeg_lock:
                    self.latest_jpeg = buffer.tobytes()

            # Optional small delay to prevent massive CPU consumption on fast local files (target ~30fps)
            elapsed = time.time() - start_time
            sleep_time = max(0, (1.0 / 30.0) - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

        if cap is not None:
            cap.release()
        logger.info("Ingestion worker gracefully terminated.")

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Main entry point. Accepts a raw BGR frame, runs all detection
        and violation checks, returns an annotated BGR frame.
        """
        self._frame_index += 1
        frame_idx = self._frame_index

        # Step 1 — Resize to fixed processing canvas
        canvas = cv2.resize(frame, (PROCESS_W, PROCESS_H))

        # Step 2 — YOLO inference with ByteTrack persistent tracking
        results = self.model.track(
            canvas,
            persist=True,
            conf=self.conf_thresh,
            iou=0.45,
            verbose=False,
        )

        if results is None or len(results) == 0:
            return self._draw_roi(canvas)

        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            return self._draw_roi(canvas)

        boxes       = result.boxes.xyxy.cpu().numpy()       # (N, 4) [x1,y1,x2,y2]
        class_ids   = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()

        # Track IDs may be None for untracked frames
        track_ids_raw = result.boxes.id
        track_ids = (
            track_ids_raw.cpu().numpy().astype(int)
            if track_ids_raw is not None
            else np.full(len(boxes), -1, dtype=int)
        )

        # Step 3 — Update per-track coordinate histories
        self._update_trajectories(boxes, class_ids, track_ids, frame_idx)

        # Step 4 — Emergency vehicle detection (runs before everything else)
        self._check_emergency_override(class_ids, track_ids, canvas, frame_idx)

        # Step 5 — Run violation detectors only when no emergency lock is active
        if not self.emergency_active:
            self._check_wrong_lane(boxes, class_ids, track_ids, canvas)
            self._check_triple_riding(boxes, class_ids, track_ids, canvas)
            self._check_seatbelt_phone(boxes, class_ids, track_ids, canvas)

        # Step 6 — Adaptive signal density update
        self._update_density_signal(class_ids)

        # Step 7 — Tick per-track cooldowns and prune stale states
        self._tick_and_prune(track_ids, frame_idx)

        # Step 8 — Annotate and return
        annotated = result.plot()  # built-in ultralytics overlay
        annotated = self._draw_roi(annotated)
        annotated = self._draw_status_banner(annotated)

        return annotated

    def process_video(self, video_path: str, output_path: Optional[str] = None) -> None:
        """
        Processes a video file end-to-end. Optionally writes annotated output.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps     = cap.get(cv2.CAP_PROP_FPS) or 25.0
        orig_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer: Optional[cv2.VideoWriter] = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (PROCESS_W, PROCESS_H))

        logger.info(
            "Processing video: %s (%dx%d @ %.1f fps)", video_path, orig_w, orig_h, fps
        )

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                annotated = self.process_frame(frame)
                if writer:
                    writer.write(annotated)
        finally:
            cap.release()
            if writer:
                writer.release()
            logger.info(
                "Video processing complete. Frames processed: %d", self._frame_index
            )

    def trigger_manual_override(self, active: bool) -> None:
        """
        Allows external callers (e.g. FastAPI endpoint) to force-set emergency state.
        """
        self.emergency_active = active
        state = "ACTIVE" if active else "CLEARED"
        logger.warning("Manual emergency override: %s", state)

    # ------------------------------------------------------------------ #
    #  Step 3 — Trajectory bookkeeping                                     #
    # ------------------------------------------------------------------ #

    def _update_trajectories(
        self,
        boxes: np.ndarray,
        class_ids: np.ndarray,
        track_ids: np.ndarray,
        frame_idx: int,
    ) -> None:
        with self._state_lock:
            for i, tid in enumerate(track_ids):
                if tid < 0:
                    continue
                cx = float((boxes[i][0] + boxes[i][2]) / 2.0)
                cy = float((boxes[i][1] + boxes[i][3]) / 2.0)

                if tid not in self._track_states:
                    self._track_states[tid] = TrackState(tid)
                state = self._track_states[tid]
                state.push(cx, cy)
                state.last_seen_frame = frame_idx

    # ------------------------------------------------------------------ #
    #  Step 4 — Emergency vehicle override                                 #
    # ------------------------------------------------------------------ #

    def _check_emergency_override(
        self,
        class_ids: np.ndarray,
        track_ids: np.ndarray,
        canvas: np.ndarray,
        frame_idx: int,
    ) -> None:
        """
        Detects ambulance/emergency vehicle class in current frame.
        Locks the pipeline state while the vehicle is visible.
        Schedules a deferred reset EMERGENCY_DROPOUT_RESET frames after last sighting.
        """
        active_emergency_ids = set()

        for i, cls in enumerate(class_ids):
            if cls == CLASS_AMBULANCE:
                tid = int(track_ids[i]) if track_ids[i] >= 0 else -1
                active_emergency_ids.add(tid)

                if not self.emergency_active:
                    logger.warning(
                        "Frame %d: Emergency vehicle detected (track_id=%d). "
                        "System LOCKED. All violation checks suspended.",
                        frame_idx, tid,
                    )
                self.emergency_active = True
                self._emergency_track_ids.add(tid)
                # Reset dropout counter every frame it's visible
                self._emergency_dropout_counters[tid] = EMERGENCY_DROPOUT_RESET

        # Check for IDs that have now dropped out of frame
        departed_ids = self._emergency_track_ids - active_emergency_ids
        for tid in list(departed_ids):
            remaining = self._emergency_dropout_counters.get(tid, 0) - 1
            self._emergency_dropout_counters[tid] = remaining
            if remaining <= 0:
                # Perform cleanup for this emergency track
                self._emergency_track_ids.discard(tid)
                del self._emergency_dropout_counters[tid]
                logger.info(
                    "Frame %d: Emergency track_id=%d fully cleared (dropout=%d frames).",
                    frame_idx, tid, EMERGENCY_DROPOUT_RESET,
                )

        # Release lock only when ALL emergency IDs have cleared their dropout timers
        if (
            self.emergency_active
            and not self._emergency_track_ids
            and not self._emergency_dropout_counters
        ):
            self.emergency_active = False
            self._logged_events.clear()   # reset deduplication after emergency clears
            logger.info(
                "Frame %d: Emergency CLEARED. Pipeline state fully reset.", frame_idx
            )

    # ------------------------------------------------------------------ #
    #  Step 5a — Wrong-lane vector analysis                                #
    # ------------------------------------------------------------------ #

    def _check_wrong_lane(
        self,
        boxes: np.ndarray,
        class_ids: np.ndarray,
        track_ids: np.ndarray,
        canvas: np.ndarray,
    ) -> None:
        """
        Evaluates rolling trajectory displacement for each tracked vehicle.

        Flow logic (640×480 canvas, y increases downward):
        - 'top_to_bottom': normal = Δy > 0. Wrong lane = Δy < WRONG_LANE_DELTA_Y (i.e. < -30 px).
        - 'bottom_to_top': normal = Δy < 0. Wrong lane = Δy > abs(WRONG_LANE_DELTA_Y) (i.e. > +30 px).
        """
        with self._state_lock:
            for i, tid in enumerate(track_ids):
                if tid < 0:
                    continue

                state = self._track_states.get(tid)
                if state is None or state.wrong_lane_cooldown > 0:
                    continue

                delta_y = state.trajectory_delta_y()
                if delta_y is None:
                    continue

                # ── Point-in-ROI guard ─────────────────────────────────────
                cx = float((boxes[i][0] + boxes[i][2]) / 2.0)
                cy = float((boxes[i][1] + boxes[i][3]) / 2.0)
                if not self._point_in_roi(cx, cy):
                    continue

                violation_triggered = False
                if self.flow_direction == "top_to_bottom":
                    # Upward displacement in a downward-flow zone is wrong-lane
                    if delta_y < WRONG_LANE_DELTA_Y:
                        violation_triggered = True
                elif self.flow_direction == "bottom_to_top":
                    # Downward displacement in an upward-flow zone is wrong-lane
                    if delta_y > abs(WRONG_LANE_DELTA_Y):
                        violation_triggered = True

                if violation_triggered:
                    event_key = (tid, "WrongLane")
                    if event_key not in self._logged_events:
                        self._logged_events.add(event_key)
                        img_path = self._save_evidence_crop(
                            canvas, boxes[i], f"wronglane_tid{tid}"
                        )
                        insert_violation(
                            timestamp=self._now(),
                            violation_type="Wrong Lane",
                            registration_string=f"TRACK_{tid}",
                            confidence=float(1.0),
                            image_path=img_path,
                        )
                        logger.info(
                            "VIOLATION [WrongLane] track_id=%d delta_y=%.1f direction=%s",
                            tid, delta_y, self.flow_direction,
                        )
                    state.wrong_lane_cooldown = WRONG_LANE_COOLDOWN

    # ------------------------------------------------------------------ #
    #  Step 5b — Triple riding & head/helmet isolation                     #
    # ------------------------------------------------------------------ #

    def _check_triple_riding(
        self,
        boxes: np.ndarray,
        class_ids: np.ndarray,
        track_ids: np.ndarray,
        canvas: np.ndarray,
    ) -> None:
        """
        For each motorcycle (class 3):
          1. Count person (class 0) centroids inside the motorcycle bounding box.
          2. If count > MAX_RIDERS → log 'Triple Riding'.
          3. Extract the upper HEAD_ZONE_FRACTION of each overlapping person box
             and persist the crop for secondary helmet-verification input.
        """
        moto_indices   = np.where(class_ids == CLASS_MOTORCYCLE)[0]
        person_indices = np.where(class_ids == CLASS_PERSON)[0]

        for mi in moto_indices:
            mx1, my1, mx2, my2 = boxes[mi]
            moto_tid = int(track_ids[mi]) if track_ids[mi] >= 0 else -1

            # ── Collect overlapping persons ────────────────────────────────
            overlapping_persons: List[int] = []
            for pi in person_indices:
                pcx = float((boxes[pi][0] + boxes[pi][2]) / 2.0)
                pcy = float((boxes[pi][1] + boxes[pi][3]) / 2.0)
                if mx1 <= pcx <= mx2 and my1 <= pcy <= my2:
                    overlapping_persons.append(pi)

            rider_count = len(overlapping_persons)

            # ── Triple-riding check ────────────────────────────────────────
            if rider_count > MAX_RIDERS:
                event_key = (moto_tid, "TripleRiding")
                if event_key not in self._logged_events:
                    self._logged_events.add(event_key)
                    img_path = self._save_evidence_crop(
                        canvas, boxes[mi], f"tripleride_tid{moto_tid}"
                    )
                    insert_violation(
                        timestamp=self._now(),
                        violation_type="Triple Riding",
                        registration_string=f"TRACK_{moto_tid}",
                        confidence=float(rider_count / (MAX_RIDERS + 1)),
                        image_path=img_path,
                    )
                    logger.info(
                        "VIOLATION [TripleRiding] moto_tid=%d riders_detected=%d",
                        moto_tid, rider_count,
                    )

            # ── Head/helmet zone extraction for each overlapping person ────
            for pi in overlapping_persons:
                px1 = int(boxes[pi][0])
                py1 = int(boxes[pi][1])
                px2 = int(boxes[pi][2])
                py2 = int(boxes[pi][3])
                person_h = py2 - py1

                # Upper HEAD_ZONE_FRACTION (20%) = head region
                head_y2 = py1 + int(person_h * HEAD_ZONE_FRACTION)

                # Clamp to canvas bounds
                hx1 = max(0, px1)
                hy1 = max(0, py1)
                hx2 = min(PROCESS_W, px2)
                hy2 = min(PROCESS_H, head_y2)

                if hy2 > hy1 and hx2 > hx1:
                    head_crop = canvas[hy1:hy2, hx1:hx2]
                    person_tid = int(track_ids[pi]) if track_ids[pi] >= 0 else pi
                    # Persist the crop — downstream helmet classifier reads this dir
                    self._write_head_crop(head_crop, person_tid)

    def _write_head_crop(self, head_crop: np.ndarray, person_tid: int) -> None:
        """Persists a head-zone crop to disk for secondary helmet classifier input."""
        if head_crop.size == 0:
            return
        fname = EVIDENCE_DIR / f"headzone_tid{person_tid}_{self._frame_index:06d}.jpg"
        try:
            cv2.imwrite(str(fname), head_crop)
        except Exception as exc:
            logger.warning("Could not write head crop: %s", exc)

    # ------------------------------------------------------------------ #
    #  Step 5c — Seatbelt / phone obstruction check                        #
    # ------------------------------------------------------------------ #

    def _check_seatbelt_phone(
        self,
        boxes: np.ndarray,
        class_ids: np.ndarray,
        track_ids: np.ndarray,
        canvas: np.ndarray,
    ) -> None:
        """
        For each car/truck/bus in the frame:
          1. Slice the upper-middle subsection of the bounding box (cabin region).
          2. Compute mean brightness of the grayscale cabin crop.
          3. Run Sobel gradient magnitude analysis for internal obstruction edges.
          4. If brightness < CABIN_DARK_THRESH OR mean gradient > 35.0 →
             classify as a Seatbelt/Phone Violation.

        This heuristic is conservative by design; integrate a dedicated
        seatbelt/phone classifier model to replace the brightness+gradient gate.
        """
        for i, cls in enumerate(class_ids):
            if cls not in VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = boxes[i]
            tid    = int(track_ids[i]) if track_ids[i] >= 0 else -1
            box_w  = x2 - x1
            box_h  = y2 - y1

            # ── Cabin slice boundaries ─────────────────────────────────────
            cabin_y1 = int(y1 + box_h * CABIN_TOP_FRAC)
            cabin_y2 = int(y1 + box_h * CABIN_BOTTOM_FRAC)
            cabin_x1 = int(x1 + box_w * CABIN_LEFT_FRAC)
            cabin_x2 = int(x1 + box_w * CABIN_RIGHT_FRAC)

            # Clamp to canvas
            cabin_y1 = max(0, cabin_y1)
            cabin_y2 = min(PROCESS_H, cabin_y2)
            cabin_x1 = max(0, cabin_x1)
            cabin_x2 = min(PROCESS_W, cabin_x2)

            if cabin_y2 <= cabin_y1 or cabin_x2 <= cabin_x1:
                continue

            cabin_crop = canvas[cabin_y1:cabin_y2, cabin_x1:cabin_x2]
            if cabin_crop.size == 0:
                continue

            # ── Vector validation: mean brightness of grayscale cabin ──────
            gray_cabin  = cv2.cvtColor(cabin_crop, cv2.COLOR_BGR2GRAY)
            mean_bright = float(np.mean(gray_cabin))

            # ── Gradient-based obstruction metric ─────────────────────────
            sobelx        = cv2.Sobel(gray_cabin, cv2.CV_64F, 1, 0, ksize=3)
            sobely        = cv2.Sobel(gray_cabin, cv2.CV_64F, 0, 1, ksize=3)
            gradient_mag  = np.sqrt(sobelx**2 + sobely**2)
            mean_gradient = float(np.mean(gradient_mag))

            # Anomaly condition: dark region OR high internal gradient
            anomaly = (mean_bright < CABIN_DARK_THRESH) or (mean_gradient > 35.0)

            if anomaly:
                event_key = (tid, "SeatbeltPhone")
                if event_key not in self._logged_events:
                    self._logged_events.add(event_key)
                    img_path = self._save_evidence_crop(
                        canvas, boxes[i], f"seatbelt_tid{tid}"
                    )
                    insert_violation(
                        timestamp=self._now(),
                        violation_type="Seatbelt/Phone Violation",
                        registration_string=f"TRACK_{tid}",
                        confidence=round(mean_gradient / 100.0, 3),
                        image_path=img_path,
                    )
                    logger.info(
                        "VIOLATION [Seatbelt/Phone] tid=%d bright=%.1f gradient=%.1f",
                        tid, mean_bright, mean_gradient,
                    )

    # ------------------------------------------------------------------ #
    #  Step 6 — Adaptive signal density update                             #
    # ------------------------------------------------------------------ #

    def _update_density_signal(self, class_ids: np.ndarray) -> None:
        """
        Counts vehicles in current frame, computes density, calculates adaptive
        green time, and persists via update_signal_metrics().
        """
        vehicle_mask  = np.isin(class_ids, list(VEHICLE_CLASSES))
        vehicle_count = int(np.sum(vehicle_mask))
        density       = float(vehicle_count)
        green_time    = BASE_GREEN_TIME + vehicle_count * DENSITY_PER_SECOND

        update_signal_metrics(
            junction_name=JUNCTION_NAME,
            current_density=density,
            allocated_green_time=green_time,
        )

    # ------------------------------------------------------------------ #
    #  Step 7 — Cooldown tick & stale-state pruning                        #
    # ------------------------------------------------------------------ #

    def _tick_and_prune(self, active_track_ids: np.ndarray, frame_idx: int) -> None:
        """
        Ticks cooldown counters for all states and removes entries that have
        not been seen for more than 2× TRAJECTORY_WINDOW frames (stale IDs).
        """
        stale_threshold = TRAJECTORY_WINDOW * 2
        with self._state_lock:
            for tid, state in list(self._track_states.items()):
                state.tick_cooldown()
                if (frame_idx - state.last_seen_frame) > stale_threshold:
                    del self._track_states[tid]
                    self._logged_events = {
                        ev for ev in self._logged_events if ev[0] != tid
                    }

    # ------------------------------------------------------------------ #
    #  Utility methods                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _point_in_roi(cx: float, cy: float) -> bool:
        """
        Tests whether a centroid (cx, cy) lies inside the global ROI polygon.
        Uses OpenCV's pointPolygonTest (positive result → inside).
        """
        pt     = (float(cx), float(cy))
        result = cv2.pointPolygonTest(ROI_POLYGON, pt, measureDist=False)
        return result >= 0

    @staticmethod
    def _now() -> str:
        """Returns ISO-8601 UTC timestamp string."""
        return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    @staticmethod
    def _save_evidence_crop(
        canvas: np.ndarray,
        box: np.ndarray,
        tag: str,
    ) -> str:
        """
        Crops the detection bounding box from the canvas and saves to evidence dir.
        Returns the absolute file path string for database storage.
        """
        x1, y1, x2, y2 = (int(v) for v in box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(PROCESS_W, x2), min(PROCESS_H, y2)

        crop     = canvas[y1:y2, x1:x2] if (y2 > y1 and x2 > x1) else canvas
        ts       = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")
        filename = EVIDENCE_DIR / f"{tag}_{ts}.jpg"

        try:
            cv2.imwrite(str(filename), crop)
        except Exception as exc:
            logger.warning("Evidence crop save failed (%s): %s", filename, exc)

        return str(filename.resolve())

    def _draw_roi(self, canvas: np.ndarray) -> np.ndarray:
        """Overlays the ROI polygon boundary on the canvas."""
        cv2.polylines(
            canvas,
            [ROI_POLYGON],
            isClosed=True,
            color=(0, 215, 255),   # gold
            thickness=2,
            lineType=cv2.LINE_AA,
        )
        return canvas

    def _draw_status_banner(self, canvas: np.ndarray) -> np.ndarray:
        """Renders a status banner at the top of the frame."""
        if self.emergency_active:
            label = "!! EMERGENCY OVERRIDE ACTIVE !!"
            color = (0, 0, 220)
        else:
            label = f"SentinelVision | Frame {self._frame_index} | {self.flow_direction}"
            color = (30, 200, 30)

        cv2.rectangle(canvas, (0, 0), (PROCESS_W, 26), (20, 20, 20), -1)
        cv2.putText(
            canvas, label,
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            1,
            cv2.LINE_AA,
        )
        return canvas


# ---------------------------------------------------------------------------
# Module-level convenience runner
# ---------------------------------------------------------------------------

def run_pipeline(
    source: str,
    model_path: str = "models/yolov8n.pt",
    flow_direction: str = "top_to_bottom",
    confidence: float = 0.45,
    output_path: Optional[str] = None,
) -> None:
    """
    Convenience entry-point for running the pipeline against a file or RTSP URL.

    Parameters
    ----------
    source        : path to a video file or RTSP stream URL.
    model_path    : YOLOv8 .pt weights file.
    flow_direction: 'top_to_bottom' or 'bottom_to_top'.
    confidence    : detection confidence threshold.
    output_path   : optional path for annotated output video.
    """
    from database import init_db
    init_db()

    pipeline = VisionPipeline(
        model_path=model_path,
        flow_direction=flow_direction,
        confidence_threshold=confidence,
    )
    pipeline.process_video(source, output_path=output_path)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="SentinelVision Vision Pipeline")
    ap.add_argument("--source",  required=True,               help="Video file path or RTSP URL")
    ap.add_argument("--model",   default="models/yolov8n.pt", help="YOLO .pt weights")
    ap.add_argument("--flow",    default="top_to_bottom",     help="top_to_bottom | bottom_to_top")
    ap.add_argument("--conf",    type=float, default=0.45,    help="Detection confidence threshold")
    ap.add_argument("--output",  default=None,                help="Optional output video path")
    args = ap.parse_args()

    run_pipeline(
        source=args.source,
        model_path=args.model,
        flow_direction=args.flow,
        confidence=args.conf,
        output_path=args.output,
    )
