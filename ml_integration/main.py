"""
main.py - FastAPI Backend for ML Stress & Fatigue Detection
============================================================
RESTful API Architecture:
  /api/sessions/start   — Start a new session with mood ground truth
  /api/sessions/pause   — Pause the session (freeze tumbling window)
  /api/sessions/resume  — Resume session after pause
  /api/sessions/wake    — Resume from AFK state
  /api/sessions/end     — Graceful or interrupted session end
  /api/telemetry        — Pure data ingestion (mouse/keyboard events)
  /api/feedback         — Human-in-the-loop labelling
  /api/status/{user_id} — Query latest prediction & session info

All timestamps are expected in UTC (ms since epoch).
In-memory session buffers + PostgreSQL for persistent data.
"""

import os
import time
import logging
import asyncio
from datetime import datetime, timezone
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session as DBSession

# ML modules (same directory)
from realtime_adapter import RealTimeFeatureExtractor
from ml_pipeline import run_prediction, save_verified_data, train_local_model
import config

# Database
from database import engine, get_db, check_db_connection, Base
from db_models import User, SessionLog

# -----------------------------------------------
# LOGGING
# -----------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("backend")

# -----------------------------------------------
# CONSTANTS
# -----------------------------------------------
CHUNK_DURATION_MS = 20 * 1000  # exactly 20 seconds in milliseconds


# -----------------------------------------------
# PYDANTIC MODELS — Separated by endpoint
# -----------------------------------------------

# --- Shared event models ---
class MouseEvent(BaseModel):
    x: float
    y: float
    type: str  # 'move' | 'click' | 'mousedown' | 'mouseup' | 'scroll'
    time: float  # ms since epoch (UTC)


class KeyboardEvent(BaseModel):
    key: str
    type: str  # 'press' | 'release'
    time: float  # ms since epoch (UTC)


class WindowEvent(BaseModel):
    app_name: str
    time: float  # ms since epoch (UTC)


# --- Session endpoints ---
class SessionStartPayload(BaseModel):
    """POST /api/sessions/start"""
    session_id: str = Field(..., min_length=1)
    user_id: str = Field(..., min_length=1)
    initial_stress: Optional[str] = None
    initial_fatigue: Optional[str] = None


class SessionSignalPayload(BaseModel):
    """POST /api/sessions/pause, /resume, /wake"""
    session_id: str = Field(..., min_length=1)
    user_id: str = Field(..., min_length=1)
    afk_started_at: Optional[str] = None  # ISO timestamp (only for AFK transitions)


class SessionEndPayload(BaseModel):
    """POST /api/sessions/end — graceful close or interrupted (beforeunload)"""
    session_id: str = Field(..., min_length=1)
    user_id: str = Field(..., min_length=1)
    reason: str = "interrupted"  # 'interrupted' | 'manual'
    mouse_events: list[MouseEvent] = []
    keyboard_events: list[KeyboardEvent] = []
    afk_started_at: Optional[str] = None


# --- Telemetry data endpoint ---
class TelemetryPayload(BaseModel):
    """POST /api/telemetry — pure event data ingestion"""
    session_id: str = Field(..., min_length=1)
    user_id: str = Field(..., min_length=1)
    first_event_timestamp: Optional[str] = None
    last_event_timestamp: Optional[str] = None
    mouse_events: list[MouseEvent] = []
    keyboard_events: list[KeyboardEvent] = []
    window_events: list[WindowEvent] = []


# --- Feedback endpoint ---
class FeedbackPayload(BaseModel):
    """POST /api/feedback — human-in-the-loop labelling"""
    session_id: str = Field(..., min_length=1)
    user_id: str = Field(..., min_length=1)
    stress_label: str  # 'Stressed' | 'Not_Stressed'
    fatigue_label: str  # 'Fatigued' | 'Not_Fatigued'


# -----------------------------------------------
# IN-MEMORY STATE MANAGEMENT
# -----------------------------------------------
class UserSession:
    """Holds buffered events and ML state for a single user."""

    def __init__(self, user_id: str):
        self.user_id = user_id
        # Raw event buffers (accumulate between chunk flushes)
        self.mouse_buffer: list[dict] = []
        self.keyboard_buffer: list[dict] = []
        self.window_buffer: list[dict] = []
        # Timestamp of the current chunk window start (ms UTC)
        self.chunk_start_ms: Optional[float] = None
        # Feature extractor keeps rolling history across chunks
        self.feature_extractor = RealTimeFeatureExtractor()
        # Latest prediction result
        self.last_prediction: Optional[dict] = None
        # Latest extracted features (needed for feedback -> save_verified_data)
        self.last_features: Optional[dict] = None
        # Total verified feedback count (for training trigger)
        self.feedback_count: int = 0
        # Session state tracking
        self.state: str = "IDLE"  # IDLE | ACTIVE | AFK | PAUSED
        self.initial_stress: Optional[str] = None
        self.initial_fatigue: Optional[str] = None
        self.last_activity_ms: float = time.time() * 1000

        # --- Session Metrics (Cognitive Snapshot) ---
        self.session_start_time: float = time.time()
        self.total_mouse_events: int = 0
        self.total_keyboard_events: int = 0
        self.total_chunks_processed: int = 0
        self.stress_predictions: list[str] = []
        self.fatigue_predictions: list[str] = []
        self.stress_probabilities: list[float] = []
        self.fatigue_probabilities: list[float] = []
        self.state_timeline: list[dict] = []
        self.training_triggered_count: int = 0
        self.peak_stress_time: Optional[str] = None
        self.peak_stress_value: float = 0.0

    def update_activity(self):
        self.last_activity_ms = time.time() * 1000

    def reset_buffers(self):
        self.mouse_buffer.clear()
        self.keyboard_buffer.clear()
        self.window_buffer.clear()


# Global session store: user_id -> UserSession
sessions: dict[str, UserSession] = {}


def get_or_create_session(session_id: str, user_id: str) -> UserSession:
    if session_id not in sessions:
        sessions[session_id] = UserSession(user_id)
        logger.info(f"[SESSION] New session created for user '{user_id}' with id '{session_id}'")
    
    session = sessions[session_id]
    session.update_activity()
    return session


# -----------------------------------------------
# HELPERS
# -----------------------------------------------
def get_daylight_flags() -> dict:
    """Return daylight one-hot flags based on current UTC hour."""
    hour = datetime.now(timezone.utc).hour
    if 6 <= hour < 12:
        return {"daylight_morning": 1, "daylight_afternoon": 0, "daylight_evening": 0}
    elif 12 <= hour < 18:
        return {"daylight_morning": 0, "daylight_afternoon": 1, "daylight_evening": 0}
    else:
        return {"daylight_morning": 0, "daylight_afternoon": 0, "daylight_evening": 1}


def is_idle(session: UserSession) -> bool:
    """A chunk is idle if there are zero events across all three channels."""
    return (
        len(session.mouse_buffer) == 0
        and len(session.keyboard_buffer) == 0
        and len(session.window_buffer) == 0
    )


def process_chunk(session: UserSession) -> Optional[dict]:
    """
    Take the current buffer, run feature extraction + ML prediction,
    then flush the buffer and return the prediction result (or None if idle).
    """
    if is_idle(session):
        logger.info(f"[IDLE] User '{session.user_id}' chunk is idle (AFK). Skipping ML prediction.")
        session.reset_buffers()
        session.chunk_start_ms = None
        return None

    # Build the payload exactly as realtime_adapter expects
    payload = {
        "mouse_events": session.mouse_buffer,
        "keyboard_events": session.keyboard_buffer,
        "window_events": session.window_buffer,
    }

    # 1. Feature extraction (47 features)
    features = session.feature_extractor.process_payload(payload)

    # 2. Inject correct daylight flags based on server time
    features.update(get_daylight_flags())

    # 3. Run ML prediction (per-user model preferred, global fallback)
    prediction = run_prediction(features, user_id=session.user_id)

    # Save for later (feedback endpoint needs last_features)
    session.last_features = features
    session.last_prediction = prediction

    # --- Update session metrics for Cognitive Snapshot ---
    session.total_chunks_processed += 1
    if "predictions" in prediction:
        stress_pred = prediction["predictions"].get("stress_val")
        fatigue_pred = prediction["predictions"].get("fatigue_val")
        if stress_pred:
            session.stress_predictions.append(stress_pred)
        if fatigue_pred:
            session.fatigue_predictions.append(fatigue_pred)
    if "probabilities" in prediction:
        stress_prob = prediction["probabilities"].get("stress_val", {}).get("Stressed", 0)
        fatigue_prob = prediction["probabilities"].get("fatigue_val", {}).get("Fatigued", 0)
        session.stress_probabilities.append(round(stress_prob, 4))
        session.fatigue_probabilities.append(round(fatigue_prob, 4))
        if stress_prob > session.peak_stress_value:
            session.peak_stress_value = stress_prob
            session.peak_stress_time = datetime.now(timezone.utc).strftime("%H:%M")

    logger.info(f"[PREDICT] User '{session.user_id}' -> {prediction}")

    # Flush
    session.reset_buffers()
    session.chunk_start_ms = None

    return prediction


# -----------------------------------------------
# FASTAPI APP
# -----------------------------------------------

async def zombie_cleanup_task():
    """Background task to remove inactive sessions (zombie GC)."""
    ZOMBIE_THRESHOLD_MS = 30 * 60 * 1000  # 30 minutes
    while True:
        await asyncio.sleep(60)  # Check every minute
        now_ms = time.time() * 1000
        zombies = []
        for sid, sess in list(sessions.items()):
            if now_ms - sess.last_activity_ms > ZOMBIE_THRESHOLD_MS:
                zombies.append(sid)
        
        for sid in zombies:
            del sessions[sid]
            logger.info(f"[GC] Removed zombie session '{sid}' due to 30m inactivity.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("  ML Backend starting up  (PostgreSQL Edition)")
    logger.info(f"  Database URL      : {config.DATABASE_URL.split('@')[-1]}")
    logger.info(f"  Global model path : {config.GLOBAL_MODEL_PATH}")
    logger.info(f"  Local model path  : {config.LOCAL_MODEL_PATH}")
    logger.info(f"  Chunk duration    : {CHUNK_DURATION_MS / 1000:.0f}s")
    logger.info("=" * 60)
    
    # --- PostgreSQL: Check connection & create tables ---
    if not check_db_connection():
        logger.error("=" * 60)
        logger.error("  FATAL: PostgreSQL is not reachable!")
        logger.error("  Make sure Docker is running: docker compose up -d")
        logger.error("=" * 60)
        raise RuntimeError("PostgreSQL connection failed. Run 'docker compose up -d' first.")
    
    Base.metadata.create_all(bind=engine)
    logger.info("[DB] All tables created/verified successfully.")
    
    # Start GC task
    gc_task = asyncio.create_task(zombie_cleanup_task())
    
    yield
    
    gc_task.cancel()
    logger.info("ML Backend shutting down.")


app = FastAPI(
    title="Stress & Fatigue Detection Backend",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS — allow frontend dev servers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- STATIC FILES (Frontend) ----
frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


# ===============================================
# SESSION ENDPOINTS — State transitions only
# ===============================================

@app.get("/", include_in_schema=False)
def serve_frontend():
    return FileResponse(os.path.join(frontend_dir, "index.html"))


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.post("/api/sessions/start")
def session_start(payload: SessionStartPayload, db: DBSession = Depends(get_db)):
    """
    IDLE → ACTIVE
    Creates or resets user session, records initial mood (ground truth).
    Registers user and session in PostgreSQL for tracking.
    """
    session = get_or_create_session(payload.session_id, payload.user_id)
    session.reset_buffers()
    session.chunk_start_ms = None
    session.state = "ACTIVE"
    session.initial_stress = payload.initial_stress
    session.initial_fatigue = payload.initial_fatigue
    session.feedback_count = 0
    session.session_start_time = time.time()
    session.state_timeline = [{"state": "ACTIVE", "timestamp": time.time()}]
    session.last_prediction = None
    session.last_features = None

    # --- Register user & session in PostgreSQL ---
    try:
        existing_user = db.query(User).filter_by(user_id=payload.user_id).first()
        if not existing_user:
            db.add(User(user_id=payload.user_id))
            db.flush()
        
        existing_session = db.query(SessionLog).filter_by(session_id=payload.session_id).first()
        if not existing_session:
            db.add(SessionLog(
                session_id=payload.session_id,
                user_id=payload.user_id,
                initial_stress=payload.initial_stress,
                initial_fatigue=payload.initial_fatigue,
            ))
        db.commit()
    except Exception as e:
        logger.warning(f"[DB] Failed to log session start (non-critical): {e}")
        db.rollback()

    logger.info(f"[SESSION_START] User '{payload.user_id}' — stress: {payload.initial_stress}, fatigue: {payload.initial_fatigue}")

    return {
        "status": "session_started",
        "user_id": payload.user_id,
        "initial_stress": payload.initial_stress,
        "initial_fatigue": payload.initial_fatigue,
    }


@app.post("/api/sessions/pause")
def session_pause(payload: SessionSignalPayload):
    """
    ACTIVE → PAUSED
    Freezes the tumbling window. Buffer is NOT flushed — preserved for resume.
    """
    session = get_or_create_session(payload.session_id, payload.user_id)
    session.state = "PAUSED"
    session.state_timeline.append({"state": "PAUSED", "timestamp": time.time()})

    logger.info(f"[PAUSED] User '{payload.user_id}' — buffer preserved: "
                f"mouse={len(session.mouse_buffer)}, keyboard={len(session.keyboard_buffer)}")

    return {
        "status": "paused",
        "user_id": payload.user_id,
        "buffer_preserved": {
            "mouse": len(session.mouse_buffer),
            "keyboard": len(session.keyboard_buffer),
        },
    }


@app.post("/api/sessions/resume")
def session_resume(payload: SessionSignalPayload):
    """
    PAUSED → ACTIVE
    Resumes the tumbling window. Buffer continues from where it left off.
    """
    session = get_or_create_session(payload.session_id, payload.user_id)
    session.state = "ACTIVE"
    session.state_timeline.append({"state": "ACTIVE", "timestamp": time.time()})

    logger.info(f"[RESUME] User '{payload.user_id}' — buffer continuing: "
                f"mouse={len(session.mouse_buffer)}, keyboard={len(session.keyboard_buffer)}")

    return {
        "status": "resumed",
        "user_id": payload.user_id,
        "buffer_sizes": {
            "mouse": len(session.mouse_buffer),
            "keyboard": len(session.keyboard_buffer),
        },
    }


@app.post("/api/sessions/wake")
def session_wake(payload: SessionSignalPayload):
    """
    AFK → ACTIVE
    User returned from inactivity. Buffer continues from where it left off.
    """
    session = get_or_create_session(payload.session_id, payload.user_id)
    session.state = "ACTIVE"
    session.state_timeline.append({"state": "ACTIVE", "timestamp": time.time()})

    logger.info(f"[WAKE] User '{payload.user_id}' returned from AFK — "
                f"buffer: mouse={len(session.mouse_buffer)}, keyboard={len(session.keyboard_buffer)}")

    return {
        "status": "wake_acknowledged",
        "user_id": payload.user_id,
        "buffer_sizes": {
            "mouse": len(session.mouse_buffer),
            "keyboard": len(session.keyboard_buffer),
        },
    }


@app.post("/api/sessions/afk")
def session_afk(payload: SessionSignalPayload):
    """
    ACTIVE → AFK
    User went inactive. Buffer is NOT flushed — preserved for wake.
    """
    session = get_or_create_session(payload.session_id, payload.user_id)
    session.state = "AFK"
    session.state_timeline.append({"state": "AFK", "timestamp": time.time()})

    logger.info(f"[AFK] User '{payload.user_id}' — afk_started_at: {payload.afk_started_at}, "
                f"buffer preserved: mouse={len(session.mouse_buffer)}, keyboard={len(session.keyboard_buffer)}")

    return {
        "status": "afk",
        "user_id": payload.user_id,
        "afk_started_at": payload.afk_started_at,
        "buffer_preserved": {
            "mouse": len(session.mouse_buffer),
            "keyboard": len(session.keyboard_buffer),
        },
    }


@app.post("/api/sessions/end")
def session_end(payload: SessionEndPayload):
    """
    ANY → IDLE
    Graceful or interrupted session end.
    If there are remaining events (e.g. from beforeunload), process them.
    Returns a comprehensive session_summary for the Cognitive Snapshot screen.
    """
    session = get_or_create_session(payload.session_id, payload.user_id)

    # Buffer any final events from beforeunload
    prediction_result = None
    if payload.mouse_events or payload.keyboard_events:
        session.total_mouse_events += len(payload.mouse_events)
        session.total_keyboard_events += len(payload.keyboard_events)
        for evt in payload.mouse_events:
            session.mouse_buffer.append(evt.model_dump())
        for evt in payload.keyboard_events:
            session.keyboard_buffer.append(evt.model_dump())
        prediction_result = process_chunk(session)

    # ---- COMPUTE SESSION SUMMARY (Cognitive Snapshot) ----
    now = time.time()
    total_duration_s = now - session.session_start_time

    # Calculate time spent in each state from timeline
    active_duration_s = 0.0
    paused_duration_s = 0.0
    afk_duration_s = 0.0

    timeline = session.state_timeline + [{"state": "IDLE", "timestamp": now}]
    for i in range(len(timeline) - 1):
        segment_duration = timeline[i + 1]["timestamp"] - timeline[i]["timestamp"]
        state = timeline[i]["state"]
        if state == "ACTIVE":
            active_duration_s += segment_duration
        elif state == "PAUSED":
            paused_duration_s += segment_duration
        elif state == "AFK":
            afk_duration_s += segment_duration

    # Stress distribution (binary ratio — kept for backwards compat)
    total_predictions = len(session.stress_predictions)
    stressed_count = session.stress_predictions.count("Stressed")
    stress_ratio = round(stressed_count / total_predictions, 2) if total_predictions > 0 else 0

    # Fatigue distribution (binary ratio)
    fatigued_count = session.fatigue_predictions.count("Fatigued")
    fatigue_ratio = round(fatigued_count / total_predictions, 2) if total_predictions > 0 else 0

    # Average probabilities (more meaningful for UI display)
    avg_stress_prob = round(sum(session.stress_probabilities) / len(session.stress_probabilities), 4) if session.stress_probabilities else 0
    avg_fatigue_prob = round(sum(session.fatigue_probabilities) / len(session.fatigue_probabilities), 4) if session.fatigue_probabilities else 0

    # Flow score: active_ratio * calm_ratio * 100 (using avg probability)
    active_ratio = active_duration_s / total_duration_s if total_duration_s > 0 else 0
    calm_ratio = 1 - avg_stress_prob
    flow_score = round(active_ratio * calm_ratio * 100)

    session.state = "IDLE"

    logger.info(f"[SESSION_END] User '{payload.user_id}' — reason: {payload.reason}, "
                f"duration: {total_duration_s:.0f}s, chunks: {session.total_chunks_processed}, "
                f"feedback: {session.feedback_count}")

    return {
        "status": "session_ended",
        "user_id": payload.user_id,
        "reason": payload.reason,
        "prediction": prediction_result,
        "session_summary": {
            # Time metrics
            "total_duration_s": round(total_duration_s),
            "active_duration_s": round(active_duration_s),
            "paused_duration_s": round(paused_duration_s),
            "afk_duration_s": round(afk_duration_s),
            # Interaction metrics
            "total_mouse_events": session.total_mouse_events,
            "total_keyboard_events": session.total_keyboard_events,
            "total_chunks_processed": session.total_chunks_processed,
            # ML prediction distributions
            "stress_ratio": stress_ratio,
            "fatigue_ratio": fatigue_ratio,
            "avg_stress_prob": avg_stress_prob,
            "avg_fatigue_prob": avg_fatigue_prob,
            "stress_probabilities": session.stress_probabilities,
            "fatigue_probabilities": session.fatigue_probabilities,
            # Flow score
            "flow_score": flow_score,
            # Peaks
            "peak_stress_time": session.peak_stress_time,
            "peak_stress_value": round(session.peak_stress_value, 2),
            # Initial ground truth (for comparison)
            "initial_stress": session.initial_stress,
            "initial_fatigue": session.initial_fatigue,
            # Model evolution
            "feedback_count": session.feedback_count,
            "training_triggered_count": session.training_triggered_count,
        },
    }


# ===============================================
# TELEMETRY ENDPOINT — Pure data ingestion
# ===============================================

@app.post("/api/telemetry")
def receive_telemetry(payload: TelemetryPayload):
    """
    Receives raw mouse/keyboard/window events from the frontend.
    Buffers them and triggers ML prediction when the tumbling window completes.
    This endpoint does NOT handle state transitions — use /api/sessions/* for that.
    """
    session = get_or_create_session(payload.session_id, payload.user_id)

    # Buffer incoming events & update session metrics
    session.total_mouse_events += len(payload.mouse_events)
    session.total_keyboard_events += len(payload.keyboard_events)
    for evt in payload.mouse_events:
        session.mouse_buffer.append(evt.model_dump())
    for evt in payload.keyboard_events:
        session.keyboard_buffer.append(evt.model_dump())
    for evt in payload.window_events:
        session.window_buffer.append(evt.model_dump())

    # Initialize chunk window start on first data arrival
    now_ms = time.time() * 1000
    if session.chunk_start_ms is None:
        session.chunk_start_ms = now_ms
        logger.info(f"[CHUNK] Started new window for user '{payload.user_id}'")

    # Check if window has elapsed
    elapsed_ms = now_ms - session.chunk_start_ms
    prediction_result = None

    if elapsed_ms >= CHUNK_DURATION_MS:
        logger.info(f"[CHUNK] Window complete for user '{payload.user_id}' ({elapsed_ms / 1000:.1f}s)")
        prediction_result = process_chunk(session)

    return {
        "status": "buffered",
        "user_id": payload.user_id,
        "buffer_sizes": {
            "mouse": len(session.mouse_buffer),
            "keyboard": len(session.keyboard_buffer),
            "window": len(session.window_buffer),
        },
        "chunk_elapsed_ms": round(elapsed_ms) if session.chunk_start_ms else 0,
        "chunk_remaining_ms": max(0, round(CHUNK_DURATION_MS - elapsed_ms)) if session.chunk_start_ms else CHUNK_DURATION_MS,
        "prediction": prediction_result,
    }


# ===============================================
# FEEDBACK ENDPOINT — Human-in-the-loop
# ===============================================

@app.post("/api/feedback")
def receive_feedback(payload: FeedbackPayload, db: DBSession = Depends(get_db)):
    """
    Receives explicit user feedback ('Stressed' / 'Not_Stressed', 'Fatigued' / 'Not_Fatigued').
    Saves the labelled telemetry row to PostgreSQL and triggers local model training when threshold is met.
    """
    session = get_or_create_session(payload.session_id, payload.user_id)

    # We need features to associate with the label
    if session.last_features is None:
        logger.info(f"[FEEDBACK] No previous ML prediction for '{payload.user_id}'. Extracting features on-the-fly.")
        
        # Check if buffers have any data at all
        if not session.mouse_buffer and not session.keyboard_buffer:
            logger.warning(f"[FEEDBACK] Empty buffers for '{payload.user_id}'. Using zero-filled features as fallback.")
            # Create a zero-filled feature dict so feedback can still be saved
            from ml_pipeline import ALL_FEATURES
            features = {f: 0.0 for f in ALL_FEATURES}
            features.update(get_daylight_flags())
            session.last_features = features
        else:
            payload_for_ext = {
                "mouse_events": session.mouse_buffer,
                "keyboard_events": session.keyboard_buffer,
                "window_events": session.window_buffer,
            }
            try:
                features = session.feature_extractor.process_payload(payload_for_ext)
                features.update(get_daylight_flags())
                session.last_features = features
            except Exception as e:
                logger.warning(f"[FEEDBACK] Feature extraction failed: {e}. Using zero-filled features as fallback.")
                from ml_pipeline import ALL_FEATURES
                features = {f: 0.0 for f in ALL_FEATURES}
                features.update(get_daylight_flags())
                session.last_features = features

    # Persist to PostgreSQL
    try:
        save_verified_data(
            db=db,
            X_row=session.last_features,
            user_id=payload.user_id,
            stress_label=payload.stress_label,
            fatigue_label=payload.fatigue_label,
            session_id=payload.session_id,
        )
    except Exception as e:
        logger.error(f"[FEEDBACK] Failed to save feedback to DB: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {str(e)}")

    session.feedback_count += 1
    logger.info(f"[FEEDBACK] Saved feedback #{session.feedback_count} for user '{payload.user_id}'")

    # Attempt local model training if enough labelled rows exist
    training_triggered = False
    try:
        if session.feedback_count >= config.TRAIN_THRESHOLD_ROWS:
            logger.info(f"[TRAIN] Threshold reached ({session.feedback_count}). Triggering local model training for '{payload.user_id}'...")
            success = train_local_model(payload.user_id)
            training_triggered = success
            if success:
                session.training_triggered_count += 1
                logger.info(f"[TRAIN] Local model for '{payload.user_id}' updated successfully!")
            else:
                logger.warning(f"[TRAIN] Local model training for '{payload.user_id}' skipped (class/data safety checks).")
    except Exception as e:
        logger.error(f"[TRAIN] Training error (non-critical): {e}")

    return {
        "status": "feedback_saved",
        "user_id": payload.user_id,
        "feedback_count": session.feedback_count,
        "training_triggered": training_triggered,
    }


# ===============================================
# STATUS ENDPOINT — Query
# ===============================================

@app.get("/api/status/{session_id}")
def get_user_status(session_id: str):
    """Returns the latest prediction and session info for a given user."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail=f"No active session '{session_id}'")

    session = sessions[session_id]
    return {
        "user_id": session.user_id,
        "state": session.state,
        "initial_stress": session.initial_stress,
        "initial_fatigue": session.initial_fatigue,
        "last_prediction": session.last_prediction,
        "feedback_count": session.feedback_count,
        "buffer_sizes": {
            "mouse": len(session.mouse_buffer),
            "keyboard": len(session.keyboard_buffer),
            "window": len(session.window_buffer),
        },
    }


# -----------------------------------------------
# ENTRY POINT
# -----------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
