"""
database.py - SQLAlchemy Database Engine & Session Factory
==========================================================
PostgreSQL connection via SQLAlchemy 2.x ORM.
Connection string is read from config.DATABASE_URL.
"""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase
import logging

import config

logger = logging.getLogger("database")

# Engine — connection pool to PostgreSQL
engine = create_engine(
    config.DATABASE_URL,
    echo=False,           # True for SQL debug logging
    pool_size=5,          # Default connection pool
    max_overflow=10,      # Extra connections when pool is full
    pool_pre_ping=True,   # Verify connections before use (prevents stale conn errors)
)

# Session factory — each call creates a new DB session
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


def get_db():
    """
    FastAPI dependency injection generator.
    Usage: db: Session = Depends(get_db)
    
    Yields a DB session and ensures cleanup after request completes.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_db_connection() -> bool:
    """
    Test database connectivity at startup.
    Returns True if connection is successful, False otherwise.
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            conn.commit()
        logger.info("[DB] PostgreSQL connection successful.")
        return True
    except Exception as e:
        logger.error(f"[DB] PostgreSQL connection FAILED: {e}")
        return False
