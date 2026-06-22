"""
migrate_csv_to_db.py - CSV → PostgreSQL Migration Script
=========================================================
One-time script to migrate existing per-user CSV data files
into the PostgreSQL telemetry_features table.

Usage:
    python migrate_csv_to_db.py
    python migrate_csv_to_db.py --verify   (only verify, don't migrate)
"""

import os
import sys
import glob
import pandas as pd
from datetime import datetime, timezone

import config
from database import engine, SessionLocal, Base
from db_models import User, TelemetryFeature

# The feature columns in CSV files (everything except user_id and labels)
LABEL_COLS = ["user_id", "stress_val_mapped", "fatigue_val_mapped"]


def discover_csv_files():
    """Find all per-user CSV files in the data directory."""
    pattern = os.path.join(config.DATA_DIR, "local_data_*.csv")
    files = glob.glob(pattern)
    
    # Also include the global pool if it has user_id column
    pool_path = config.CSV_DATA_PATH
    if os.path.exists(pool_path):
        files.append(pool_path)
    
    return files


def migrate_single_csv(csv_path: str, db) -> dict:
    """
    Migrate a single CSV file into PostgreSQL.
    Returns a stats dict with row counts.
    """
    filename = os.path.basename(csv_path)
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  [ERROR] Failed to read {filename}: {e}")
        return {"file": filename, "csv_rows": 0, "migrated": 0, "error": str(e)}
    
    if len(df) == 0:
        print(f"  [SKIP] {filename} is empty.")
        return {"file": filename, "csv_rows": 0, "migrated": 0, "error": None}
    
    # Determine user_id
    if "user_id" in df.columns:
        # CSV has user_id column
        pass
    else:
        # Extract from filename: local_data_{user_id}.csv
        user_id = filename.replace("local_data_", "").replace(".csv", "")
        df["user_id"] = user_id
    
    migrated_count = 0
    
    for _, row in df.iterrows():
        user_id = str(row.get("user_id", "unknown"))
        
        # Ensure user exists
        existing_user = db.query(User).filter_by(user_id=user_id).first()
        if not existing_user:
            db.add(User(user_id=user_id))
            db.flush()
        
        # Extract stress/fatigue labels
        stress_label = row.get("stress_val_mapped", "Not_Stressed")
        fatigue_label = row.get("fatigue_val_mapped", "Not_Fatigued")
        
        # Build features dict from all non-label columns
        features = {}
        for col in df.columns:
            if col not in LABEL_COLS:
                val = row[col]
                # Convert numpy types to Python native for JSONB
                if pd.notna(val):
                    features[col] = float(val) if isinstance(val, (int, float)) else str(val)
                else:
                    features[col] = 0.0
        
        # Insert into DB
        record = TelemetryFeature(
            user_id=user_id,
            session_id=None,  # CSV data doesn't have session references
            stress_label=str(stress_label),
            fatigue_label=str(fatigue_label),
            features=features,
        )
        db.add(record)
        migrated_count += 1
    
    db.commit()
    
    print(f"  [OK] {filename}: {len(df)} CSV rows -> {migrated_count} DB rows")
    return {"file": filename, "csv_rows": len(df), "migrated": migrated_count, "error": None}


def verify_migration(db):
    """Compare CSV row counts against DB row counts per user."""
    print("\n" + "=" * 60)
    print("  MIGRATION VERIFICATION")
    print("=" * 60)
    
    # Get DB counts per user
    from sqlalchemy import func
    db_counts = db.query(
        TelemetryFeature.user_id,
        func.count(TelemetryFeature.id)
    ).group_by(TelemetryFeature.user_id).all()
    
    print(f"\n  {'User ID':<25} {'DB Rows':>10}")
    print(f"  {'-'*25} {'-'*10}")
    
    total_db = 0
    for user_id, count in db_counts:
        print(f"  {user_id:<25} {count:>10}")
        total_db += count
    
    print(f"  {'-'*25} {'-'*10}")
    print(f"  {'TOTAL':<25} {total_db:>10}")
    
    # Total users
    user_count = db.query(User).count()
    print(f"\n  Registered users: {user_count}")
    print("=" * 60)


def main():
    verify_only = "--verify" in sys.argv
    
    print("=" * 60)
    print("  CSV -> PostgreSQL Migration Tool")
    print(f"  Database: {config.DATABASE_URL.split('@')[-1]}")
    print("=" * 60)
    
    # Ensure tables exist
    Base.metadata.create_all(bind=engine)
    
    db = SessionLocal()
    
    try:
        if verify_only:
            verify_migration(db)
            return
        
        # Check if data already exists
        existing_count = db.query(TelemetryFeature).count()
        if existing_count > 0:
            print(f"\n  [WARNING] Database already has {existing_count} telemetry rows.")
            response = input("  Continue and ADD more rows? (y/N): ").strip().lower()
            if response != "y":
                print("  Migration cancelled.")
                return
        
        # Discover CSV files
        csv_files = discover_csv_files()
        if not csv_files:
            print("\n  [INFO] No CSV files found in data/ directory. Nothing to migrate.")
            return
        
        print(f"\n  Found {len(csv_files)} CSV file(s) to migrate:\n")
        
        # Migrate each file
        results = []
        for csv_path in csv_files:
            result = migrate_single_csv(csv_path, db)
            results.append(result)
        
        # Summary
        total_csv = sum(r["csv_rows"] for r in results)
        total_migrated = sum(r["migrated"] for r in results)
        errors = [r for r in results if r["error"]]
        
        print(f"\n  Migration complete: {total_migrated}/{total_csv} rows migrated.")
        if errors:
            print(f"  [WARNING] {len(errors)} file(s) had errors.")
        
        # Verify
        verify_migration(db)
        
    finally:
        db.close()


if __name__ == "__main__":
    main()
