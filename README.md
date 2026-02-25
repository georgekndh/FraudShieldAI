🛡️ FraudShield AI

Real-time credit card fraud detection system powered by LightGBM, FastAPI, and built-in explainability.

FraudShield AI is a production-shaped fraud detection service that supports:

End-to-end model training

Automatic threshold optimization

Real-time API scoring

Feature importance inspection

Fully reproducible Demo Mode

Optional interactive Streamlit dashboard

Designed for speed, transparency, and clean deployment.

⚙️ Core Capabilities

🧠 LightGBM Fraud Model

🎯 Automatic F1 Threshold Optimization

🚀 FastAPI Real-Time Inference

📊 Feature Importance Endpoint

🔁 Hot Model Reloading

🧪 Self-Bootstrapping Demo Mode

📜 Structured Logging & Prediction Audit Log

📈 Optional SHAP & Streamlit Visualization

🧠 Tech Stack
Component	Technology
Model	LightGBM
API	FastAPI
Dashboard	Streamlit
Preprocessing	scikit-learn, pandas
Explainability	SHAP
Config	YAML
Logging	Python logging
🏗️ Project Structure
FraudShieldAI/
│
├── api/                     # FastAPI backend
│   └── app.py
│
├── src/                     # Core ML logic
│   ├── data.py
│   ├── features.py
│   └── train.py
│
├── scripts/
│   └── make_demo_data.py    # Synthetic demo data generator
│
├── config/
│   ├── training.yaml
│   └── training_demo.yaml
│
├── data/
│   └── demo/                # Auto-generated demo dataset
│
├── models/
│   └── demo/                # Demo model artifact
│
├── app_streamlit.py         # Optional dashboard
├── requirements.txt
└── .gitignore
🧩 Installation
git clone https://github.com/georgekndh/FraudShieldAI.git
cd FraudShieldAI

python -m venv .venv
.\.venv\Scripts\activate     # Windows
# source .venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
🧪 Demo Mode (Zero Setup Required)

Demo mode allows the repository to run without private datasets or pre-trained models.

It will automatically:

Generate synthetic Kaggle-style fraud data

Train a LightGBM model

Save a demo model bundle

Launch the API with that model

Step 1 – Generate Demo Data
python -m scripts.make_demo_data
Step 2 – Train Demo Model
python -m src.train \
  --config config/training_demo.yaml \
  --input data/demo/transactions_demo.parquet \
  --out models/demo/fraudshield_demo.pkl
Step 3 – Run API in Demo Mode

Windows:

set DEMO_MODE=1
uvicorn api.app:app --reload

macOS/Linux:

export DEMO_MODE=1
uvicorn api.app:app --reload

Then open:

http://localhost:8000/docs
🧮 Production / Real Mode

If using a real dataset and trained artifact:

python -m src.train \
  --config config/training.yaml \
  --input data/raw/transactions.parquet \
  --out models/fraudshield_lgbm.pkl

Then:

Windows:

set DEMO_MODE=0
set MODEL_PATH=models/fraudshield_lgbm.pkl
uvicorn api.app:app --reload

macOS/Linux:

export DEMO_MODE=0
export MODEL_PATH=models/fraudshield_lgbm.pkl
uvicorn api.app:app --reload
🌐 API Endpoints
Root

GET /

Basic service confirmation.

Health Check

GET /health

Returns:

{
  "status": "ok",
  "demo_mode": true,
  "model_loaded": true,
  "model_path": "...",
  "threshold": 0.34
}
Feature Schema

GET /schema

Returns required feature names.

Score Transaction

POST /score

Example:

{
  "data": {
    "Time": 12345,
    "Amount": 78.50,
    "V1": -1.23,
    "...": "...",
    "V28": -0.42
  }
}

Response:

{
  "fraud_probability": 0.9123,
  "flag": 1,
  "threshold": 0.34
}
Batch Scoring

POST /score-batch

Supports list of transactions.

Feature Importance

GET /model-info

Returns top features sorted by importance.

Reload Model

POST /reload-model

Reload model bundle without restarting server.

📊 Streamlit Dashboard (Optional)
streamlit run app_streamlit.py

Includes:

Fraud probability histograms

Threshold slider

SHAP explainability (if enabled)

Transaction inspection

📈 Threshold Optimization

During training:

Model evaluates multiple probability thresholds

Selects threshold maximizing F1 score

Threshold stored inside model bundle

Used automatically during inference

This ensures fraud classification balances precision and recall appropriately for imbalanced datasets.

🔐 Environment Variables
Variable	Default	Purpose
DEMO_MODE	0	Enable demo auto-bootstrap
MODEL_PATH	models/fraudshield_lgbm.pkl	Model artifact path
THRESHOLD	0.55	Override threshold
PRED_LOG	models/predictions_log.csv	Prediction audit log
DEMO_MODEL_PATH	models/demo/fraudshield_demo.pkl	Demo model path
DEMO_DATA_PATH	data/demo/transactions_demo.parquet	Demo dataset path
🧠 Design Philosophy

FraudShield AI was structured to reflect production ML patterns:

Environment-aware configuration

Deterministic artifact loading

Clear schema validation

Transparent model behavior

No reliance on private committed data

Reproducible demo environment

🚀 Status

FraudShield AI supports:

Local development

Demo deployments

Recruiter-friendly evaluation

Extension into production systems

Dockerization and monitoring integrations can be added as next steps.