import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data Paths
DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_DATA_PATH = os.path.join(DATA_DIR, "local_veri_havuzu.csv")
DUMMY_DATA_PATH = os.path.join(DATA_DIR, "dummy_veri_stressli.csv")

# Model Paths
MODELS_DIR = os.path.join(BASE_DIR, "models")
GLOBAL_MODEL_PATH = os.path.join(MODELS_DIR, "global_base_model.pkl")
LOCAL_MODEL_PATH = os.path.join(MODELS_DIR, "user_model.pkl")

# ML Thresholds & Security Limits
TRAIN_THRESHOLD_ROWS = 2   # 50 rows for fast testing/presentation. In prod to be 200.
MIN_SAMPLES_PER_CLASS = 5   # Min samples for each class to avoid single-class ValueError

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Database
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://ml_user:ml_pass_2026@localhost:5433/ml_stress_db"
)


# -----------------------------------------------
# PER-USER DYNAMIC PATH HELPERS
# -----------------------------------------------
def get_user_model_path(user_id: str) -> str:
    """Returns the per-user local model path: models/user_model_{user_id}.pkl"""
    safe_id = user_id.replace(" ", "_").replace("/", "_").replace("\\", "_")
    return os.path.join(MODELS_DIR, f"user_model_{safe_id}.pkl")


def get_user_csv_path(user_id: str) -> str:
    """Returns the per-user data CSV path: data/local_data_{user_id}.csv"""
    safe_id = user_id.replace(" ", "_").replace("/", "_").replace("\\", "_")
    return os.path.join(DATA_DIR, f"local_data_{safe_id}.csv")
