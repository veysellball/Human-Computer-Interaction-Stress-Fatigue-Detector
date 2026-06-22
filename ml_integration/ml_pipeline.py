import os
import shutil
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import balanced_accuracy_score
import config

# Database imports
from sqlalchemy.orm import Session
from database import engine
from db_models import TelemetryFeature, User, MlModel

warnings.filterwarnings('ignore')

# -----------------------------------------------
# CONFIGURATION & FEATURE DEFINITIONS
# -----------------------------------------------

# Mapped features to be used in ML (using only float/int metrics)
ALL_FEATURES = [
    'typing_rate_kps', 'inter_key_latency_mean_ms',
    'key_dwell_time_mean_ms', 'key_dwell_time_std_ms', 'backspace_count',
    'long_pause_count', 'special_key_count', 'mouse_move_count',
    'mouse_avg_speed_px_s', 'mouse_std_speed_px_s', 'mouse_idle_time_ms',
    'mouse_click_count', 'mouse_path_length_px', 'window_switch_count',
    'activity_ratio', 'mouse_speed_cv', 'typing_burst_count',
    'idle_to_active_ratio', 'click_per_move_ratio',
    'mouse_path_straightness', 'backspace_error_ratio', 'unique_app_count',
    'daylight_morning', 'daylight_afternoon', 'daylight_evening',
    'app_devenv', 'app_explorer', 'app_opera', 'app_other', 'app_skype',
    'typing_rate_kps_delta', 'typing_rate_kps_rmean3', 'typing_rate_kps_rstd3',
    'inter_key_latency_mean_ms_delta', 'inter_key_latency_mean_ms_rmean3', 'inter_key_latency_mean_ms_rstd3',
    'key_dwell_time_mean_ms_delta', 'key_dwell_time_mean_ms_rmean3', 'key_dwell_time_mean_ms_rstd3',
    'mouse_avg_speed_px_s_delta', 'mouse_avg_speed_px_s_rmean3', 'mouse_avg_speed_px_s_rstd3',
    'mouse_idle_time_ms_delta', 'mouse_idle_time_ms_rmean3', 'mouse_idle_time_ms_rstd3',
    'activity_ratio_delta', 'activity_ratio_rmean3', 'activity_ratio_rstd3'
]

LABEL_MAPS = {
    'stress_val': {
        'F_Great':     'Not_Stressed',
        'F_Good':      'Not_Stressed',
        'Neutral':     'Not_Stressed',
        'S_Stressed':  'Stressed',
        'V_Stressed':  'Stressed',
    },
    'fatigue_val': {
        'No':          'Not_Fatigued',
        'Low':         'Not_Fatigued',
        'Avg':         'Fatigued',
        'Below_Avg':   'Fatigued',
        'Above_Avg':   'Fatigued',
        'V_High':      'Fatigued',
    }
}

# -----------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------

def apply_label_mapping(df):
    """Maps the 5-class target labels to our simplified classes."""
    for col, mapping in LABEL_MAPS.items():
        if col in df.columns:
            new_col = f"{col}_mapped"
            df[new_col] = df[col].map(mapping)
    return df

def build_rf_pipeline():
    """Initializes a new Random Forest pipeline with MultiOutput support."""
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', MultiOutputClassifier(
            RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        ))
    ])

# -----------------------------------------------
# CORE ML OPERATIONS
# -----------------------------------------------

def train_global_model(source_csv_path="features_v5.csv"):
    """
    Trains the global base model from the overarching dataset PLUS any dynamically
    accumulated user data from the local data pool. This ensures continuous growth!
    """
    if not os.path.exists(source_csv_path):
        print(f"[ERROR] Global dataset {source_csv_path} not found.")
        return False
        
    # 1. Load Original Base Dataset (Safe Dataset)
    df_base = pd.read_csv(source_csv_path)
    df_base = apply_label_mapping(df_base)
    
    # --- PHASE 2 CAUTION: DATA POISONING RISK ---
    # Sizin de dediginiz gibi, UI'dan yanlis/hatali scaling ile koordinat veya sure vs gelirse 
    # model tamamen yanlis seyler ogrenecektir. Entegrasyonlar stabil olmadan "auto-retrain" yapmamali.
    # Bu yuzden bu alt kismi simdilik COMMENT haline getirdik (Devre disi).
    #
    # if os.path.exists(config.CSV_DATA_PATH):
    #     df_new = pd.read_csv(config.CSV_DATA_PATH)
    #     df = pd.concat([df_base, df_new], ignore_index=True)
    #     print(f"[INFO] Merged {len(df_base)} base rows with {len(df_new)} newly collected user rows!")
    # else:
    #     df = df_base
    
    df = df_base # -> Sadece kendi orjinal ve guvenilir datamizla model egit.
    print(f"[INFO] PROTOTYPE MODE: Training only on the golden base dataset.")
    
    target_cols = ['stress_val_mapped', 'fatigue_val_mapped']
    # Drop rows missing ANY target label
    df = df.dropna(subset=target_cols)
    
    # Select available features
    features_to_use = [f for f in ALL_FEATURES if f in df.columns]
    
    X = df[features_to_use]
    y = df[target_cols]
    
    print(f"Training Global Model with {len(X)} samples using {len(features_to_use)} features for targets: {target_cols}...")
    
    pipe = build_rf_pipeline()
    pipe.fit(X, y)
    
    # --- AUTOMATIC MODEL BACKUP ---
    if os.path.exists(config.GLOBAL_MODEL_PATH):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = config.GLOBAL_MODEL_PATH.replace(".pkl", f"_backup_{timestamp}.pkl")
        shutil.copy(config.GLOBAL_MODEL_PATH, backup_path)
        print(f"[BACKUP] Existing global model safely backed up to {backup_path}")
        
    joblib.dump({"pipeline": pipe, "features": features_to_use, "targets": target_cols}, config.GLOBAL_MODEL_PATH)
    print(f"[OK] Global Model saved to {config.GLOBAL_MODEL_PATH}")
    return True


def save_verified_data(db: Session, X_row: dict, user_id: str, stress_label: str, fatigue_label: str, session_id: str = None):
    """
    Called by backend when a user explicitly gives feedback (e.g., "I feel Very Stressed right now!").
    Saves the corresponding telemetry features alongside the TRUE labels to PostgreSQL.
    
    Args:
        db: SQLAlchemy session (injected from FastAPI Depends)
        X_row: Feature dictionary (53 ML features)
        user_id: User identifier
        stress_label: 'Stressed' | 'Not_Stressed'
        fatigue_label: 'Fatigued' | 'Not_Fatigued'
        session_id: Optional session reference
    """
    # Ensure user exists in DB
    existing_user = db.query(User).filter_by(user_id=user_id).first()
    if not existing_user:
        db.add(User(user_id=user_id))
        db.flush()  # Flush to satisfy FK constraint before inserting telemetry
    
    # Sanitize numpy types to native Python types for JSONB serialization
    sanitized_features = {}
    for k, v in X_row.items():
        if hasattr(v, 'item'):  # numpy scalar (int64, float64, etc.)
            sanitized_features[k] = v.item()
        elif isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            sanitized_features[k] = 0.0
        else:
            sanitized_features[k] = v
    
    # Create telemetry feature record with JSONB features
    record = TelemetryFeature(
        user_id=user_id,
        session_id=session_id,
        stress_label=stress_label,
        fatigue_label=fatigue_label,
        features=sanitized_features,  # Automatically serialized as JSONB by PostgreSQL
    )
    db.add(record)
    db.commit()
    
    print(f"[DATA POOL] Successfully saved 1 new labeled row for '{user_id}' -> PostgreSQL")
    return True

def train_local_model(user_id: str):
    """
    Called by backend when enough rows are accumulated for a specific user.
    Reads labelled feature data from PostgreSQL, checks class counts,
    and trains a per-user model file (user_model_{user_id}.pkl).
    """
    model_path = config.get_user_model_path(user_id)
    
    # Query labelled features from PostgreSQL
    try:
        query = "SELECT features, stress_label, fatigue_label FROM telemetry_features WHERE user_id = %(uid)s"
        df_raw = pd.read_sql(query, engine, params={"uid": user_id})
    except Exception as e:
        print(f"[ERROR] Failed to query DB for '{user_id}': {e}")
        return False
    
    if len(df_raw) == 0:
        print(f"[ERROR] No data found in DB for '{user_id}'")
        return False
    
    if len(df_raw) < config.TRAIN_THRESHOLD_ROWS:
        print(f"[INFO] Not enough rows yet for '{user_id}' ({len(df_raw)}/{config.TRAIN_THRESHOLD_ROWS})")
        return False
    
    # Expand JSONB features column into flat DataFrame
    features_df = pd.json_normalize(df_raw["features"])
    features_df["stress_val_mapped"] = df_raw["stress_label"].values
    features_df["fatigue_val_mapped"] = df_raw["fatigue_label"].values
    
    df = features_df
    
    target_cols = ["stress_val_mapped", "fatigue_val_mapped"]
    
    # Drop NaNs from targets
    df = df.dropna(subset=target_cols)
    if len(df) < config.TRAIN_THRESHOLD_ROWS:
        print(f"[WARN] Not enough valid rows after dropping NaNs ({len(df)}).")
        return False
    
    # Check class safety per target
    for tcol in target_cols:
        class_counts = df[tcol].value_counts()
        if len(class_counts) < 2:
            print(f"[WARN] Target {tcol} has only one class ({class_counts.index[0]}). Skipping local training to prevent err.")
            return False
        if any(count < config.MIN_SAMPLES_PER_CLASS for count in class_counts.values):
            print(f"[WARN] Target {tcol} classes don't meet min sample threshold. Distribution: {class_counts.to_dict()}. Skipping.")
            return False
        
    features_to_use = [f for f in ALL_FEATURES if f in df.columns]
    
    X = df[features_to_use]
    y = df[target_cols]
    
    print(f"Training Per-User Model for '{user_id}' on {len(X)} rows for targets {target_cols}...")
    
    pipe = build_rf_pipeline()
    pipe.fit(X, y)
    
    # Save model to filesystem
    joblib.dump({"pipeline": pipe, "features": features_to_use, "targets": target_cols}, model_path)
    print(f"[OK] Per-user model for '{user_id}' saved at {model_path}")
    
    # Track model metadata in DB
    try:
        from database import SessionLocal
        db = SessionLocal()
        # Deactivate previous models for this user
        db.query(MlModel).filter_by(user_id=user_id, is_active=True).update({"is_active": False})
        # Insert new model record
        db.add(MlModel(
            user_id=user_id,
            model_path=model_path,
            training_rows=len(X),
            is_active=True,
        ))
        db.commit()
        db.close()
    except Exception as e:
        print(f"[WARN] Model metadata save failed (non-critical): {e}")
    
    return True

def run_prediction(X_row: dict, user_id: str = None) -> dict:
    """
    Runs prediction on a single row (passed as dictionary mapping feature names to values).
    Prefers per-user local model over global model.
    
    Priority: user_model_{user_id}.pkl > global_base_model.pkl
    """
    # Determine which model to use (per-user first, then global fallback)
    is_local = False
    if user_id:
        user_model_path = config.get_user_model_path(user_id)
        if os.path.exists(user_model_path):
            model_path = user_model_path
            is_local = True
        else:
            model_path = config.GLOBAL_MODEL_PATH
    else:
        model_path = config.GLOBAL_MODEL_PATH
    
    if not os.path.exists(model_path):
        return {"error": f"No model file found (checked {model_path})"}
        
    model_data = joblib.load(model_path)
    pipe = model_data["pipeline"]
    features_to_use = model_data["features"]
    target_cols = model_data.get("targets", ['stress_val', 'fatigue_val'])
    
    # Build dataframe for exactly one row
    X_df = pd.DataFrame([X_row], columns=features_to_use)
    # Fill missing with zeros/nans for robustness
    for f in features_to_use:
        if f not in X_row:
            X_df[f] = 0.0
            
    try:
        prediction_array = pipe.predict(X_df)[0]
        # predict_proba returns a list of probability arrays (one for each target)
        probas_list = pipe.predict_proba(X_df)
        
        predictions = {}
        probabilities = {}
        
        estimators = pipe.named_steps['model'].estimators_
        
        for i, target_name in enumerate(target_cols):
            clean_name = target_name.replace("_mapped", "")
            predictions[clean_name] = prediction_array[i]
            
            classes = estimators[i].classes_
            class_probs = probas_list[i][0]
            probabilities[clean_name] = dict(zip(classes, class_probs))
            
        return {
            "predictions": predictions,
            "probabilities": probabilities,
            "model_used": f"LOCAL ({user_id})" if is_local else "GLOBAL"
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    train_global_model()
