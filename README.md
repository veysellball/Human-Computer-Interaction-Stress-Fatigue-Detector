# ML Stress & Fatigue Detection System

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=flat&logo=postgresql&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)

> Real-time stress and fatigue detection using keyboard, mouse, and window telemetry — personalized per user via a human-in-the-loop feedback loop.

This system collects behavioral telemetry in 20-second windows from a user's desktop, extracts 47 numerical features, and predicts **Stress** and **Fatigue** levels in real time. It ships a global base model out of the box and automatically trains a personal model for each user as they provide feedback — no manual labeling pipeline required.

## Table of Contents

- [About](#about)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)
- [Authors](#authors)

## About

The system started as a global-model approach (one shared Random Forest trained on a pooled dataset) and evolved into a **per-user model architecture** where each user's behavioral baseline is learned individually. This matters because mouse speed or typing cadence under stress differs significantly between people.

When a user reports their current state via the UI ("I'm stressed right now"), the labeled feature vector is persisted to PostgreSQL. Once enough labeled rows accumulate, the backend automatically retrains that user's personal model — no manual intervention needed. Until then, predictions fall back to the shared global model.

## Features

- **Real-time inference** — predictions returned in under 50 ms per 20-second telemetry chunk.
- **Per-user model isolation** — each user gets their own `RandomForestClassifier` stored as a `.pkl` file; no cross-user data leakage.
- **Human-in-the-loop retraining** — the model self-updates as users submit feedback labels.
- **Automatic fallback** — falls back to the global base model until a user's personal model is ready.
- **Session lifecycle management** — Start, Pause, Resume, Wake (AFK), and End session states with full logging.
- **Cognitive session snapshot** — on session end, the API returns a summary: stress distribution, peak stress timestamps, and a Flow Score.
- **47-feature extraction** — mouse velocity, click rate, backspace error rate, active/passive time ratio, rolling means, and more.

## Tech Stack

- **Backend:** Python 3.11, FastAPI, Uvicorn
- **Database:** PostgreSQL 16 (Docker), SQLAlchemy ORM
- **Machine Learning:** scikit-learn (`MultiOutputClassifier` + `RandomForestClassifier`), pandas, NumPy, joblib
- **Frontend:** Vanilla HTML/CSS/JS served as FastAPI static files (`frontend/`)
- **Infrastructure:** Docker Compose

## Project Structure

```
ml_integration/
├── main.py               # FastAPI server — all REST endpoints
├── ml_pipeline.py        # Train global model, train per-user model, run prediction
├── realtime_adapter.py   # Raw telemetry → 47 numerical features (RealTimeFeatureExtractor)
├── db_models.py          # SQLAlchemy table definitions (users, sessions_log, telemetry_features, ml_models)
├── database.py           # DB engine and session factory
├── config.py             # Constants: paths, thresholds, DB URL
├── docker-compose.yml    # Spins up PostgreSQL (port 5433)
├── requirements.txt
├── train_final.py        # One-off script to train / retrain the global base model
├── migrate_csv_to_db.py  # Migrates legacy CSV data into PostgreSQL
├── test_pipeline.py      # Pipeline unit tests
├── test_telemetry_accuracy.py
├── data/                 # Per-user CSV snapshots (legacy) and pooled data
├── models/               # Serialized .pkl model files
│   ├── global_base_model.pkl
│   └── user_model_<user_id>.pkl
└── frontend/
    ├── index.html
    ├── app.jsx
    └── style.css
```

## Getting Started

### Prerequisites

- Python 3.11+
- Docker & Docker Compose

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/your-repo.git
cd your-repo/ml_integration

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Start the PostgreSQL database
docker-compose up -d

# 4. (Optional) Train the global base model
python train_final.py
```

The database runs on **port 5433** (mapped from the container's 5432) to avoid conflicts with a local Postgres instance.

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql://ml_user:ml_pass_2026@localhost:5433/ml_stress_db` | PostgreSQL connection string |

## Usage

```bash
# Start the API server
uvicorn main:app --reload
```

Open [http://localhost:8000](http://localhost:8000) in your browser to access the web UI.

The interactive API docs are available at [http://localhost:8000/docs](http://localhost:8000/docs).

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/sessions/start` | Start a new session with baseline mood |
| `POST` | `/api/sessions/pause` | Pause the active session |
| `POST` | `/api/sessions/resume` | Resume a paused session |
| `POST` | `/api/sessions/wake` | Resume from AFK state |
| `POST` | `/api/sessions/end` | End session and return cognitive snapshot |
| `POST` | `/api/telemetry` | Ingest a 20-second telemetry chunk |
| `POST` | `/api/feedback` | Submit a human label for the current state |
| `GET`  | `/api/status/{user_id}` | Get the latest prediction and session info |

All timestamps are expected as UTC milliseconds since epoch.

## Contributing

Contributions are welcome. Please open an issue first to discuss what you'd like to change before submitting a pull request.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

### Known areas for improvement

- `TRAIN_THRESHOLD_ROWS` in `config.py` is currently set to `2` for testing — raise to `200` for production.
- WebSocket-based streaming would replace the current polling approach for lower latency.
- Alembic migrations are listed in `requirements.txt` but not yet wired up — schema changes are manual.

## License

<!-- TODO: Choose a license at https://choosealicense.com and add a LICENSE file. -->
No license has been specified yet. All rights reserved until a license is added.

## Authors

- **Veysel** — project owner
