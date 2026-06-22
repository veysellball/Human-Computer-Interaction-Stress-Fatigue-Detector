"""
db_models.py - SQLAlchemy ORM Table Definitions
================================================
Tables:
  - users              : Registered users
  - sessions_log       : Session tracking (start/end/state)
  - telemetry_features : Labelled feature rows (replaces CSV files)
  - ml_models          : Model file metadata & training history
"""

from datetime import datetime, timezone
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey, Index
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from database import Base


class User(Base):
    """Registered users — maps frontend user_id to DB record."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(100), unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    telemetry = relationship("TelemetryFeature", back_populates="user", lazy="dynamic")
    sessions = relationship("SessionLog", back_populates="user", lazy="dynamic")
    models = relationship("MlModel", back_populates="user", lazy="dynamic")

    def __repr__(self):
        return f"<User(user_id='{self.user_id}')>"


class SessionLog(Base):
    """Session tracking — records start/end and initial ground truth."""
    __tablename__ = "sessions_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(200), unique=True, nullable=False, index=True)
    user_id = Column(String(100), ForeignKey("users.user_id"), nullable=False)
    initial_stress = Column(String(50), nullable=True)
    initial_fatigue = Column(String(50), nullable=True)
    started_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    ended_at = Column(DateTime, nullable=True)
    end_reason = Column(String(50), nullable=True)  # 'manual' | 'interrupted'

    # Relationships
    user = relationship("User", back_populates="sessions")

    def __repr__(self):
        return f"<SessionLog(session_id='{self.session_id}', user='{self.user_id}')>"


class TelemetryFeature(Base):
    """
    Labelled telemetry feature rows — replaces per-user CSV files.
    Features are stored as PostgreSQL JSONB for schema flexibility.
    """
    __tablename__ = "telemetry_features"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(100), ForeignKey("users.user_id"), nullable=False)
    session_id = Column(String(200), nullable=True)
    recorded_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    stress_label = Column(String(20), nullable=False)    # 'Stressed' | 'Not_Stressed'
    fatigue_label = Column(String(20), nullable=False)   # 'Fatigued' | 'Not_Fatigued'
    features = Column(JSONB, nullable=False)              # 53 ML features as key-value

    # Relationships
    user = relationship("User", back_populates="telemetry")

    # Composite index for fast per-user queries (used by train_local_model)
    __table_args__ = (
        Index("ix_telemetry_user_id", "user_id"),
        Index("ix_telemetry_recorded_at", "recorded_at"),
        Index("ix_telemetry_user_time", "user_id", "recorded_at"),
    )

    def __repr__(self):
        return f"<TelemetryFeature(user='{self.user_id}', stress='{self.stress_label}')>"


class MlModel(Base):
    """Model file metadata — tracks which model is active for each user."""
    __tablename__ = "ml_models"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(100), ForeignKey("users.user_id"), nullable=True)  # NULL = global
    model_path = Column(String(500), nullable=False)
    trained_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    training_rows = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)

    # Relationships
    user = relationship("User", back_populates="models")

    def __repr__(self):
        return f"<MlModel(user='{self.user_id}', path='{self.model_path}', active={self.is_active})>"
