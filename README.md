# 🛡️ FraudShield AI

### Real-Time Credit Card Fraud Detection  
Built with **LightGBM · FastAPI · Scikit-Learn · SHAP**

---

FraudShield AI is a production-structured fraud detection service that supports:

- End-to-end model training  
- Automatic F1-optimized threshold selection  
- Real-time API inference  
- Feature importance inspection  
- Fully reproducible **Demo Mode**  
- Optional Streamlit dashboard with explainability  

Designed for **speed, transparency, and clean deployment.**

---

# 🚀 Core Capabilities

### 🧠 LightGBM Fraud Model
- Gradient boosting classifier
- Handles imbalanced fraud data
- Tunable hyperparameters via YAML config

### 🎯 Automatic Threshold Optimization
During training:
- Multiple probability thresholds are evaluated
- Best threshold selected via **F1 score**
- Threshold stored inside the model bundle

### ⚡ Real-Time API Scoring
- FastAPI backend
- Single & batch transaction scoring
- Live fraud probability + binary flag

### 📊 Feature Importance Endpoint
- Extracts LightGBM feature importances
- Exposed via `/model-info`
- Sorted top-K features returned

### 🔁 Hot Model Reload
- Reload model without restarting server
- Production-friendly workflow

### 🧪 Demo Mode (Reproducible)
- Generates synthetic Kaggle-style data
- Trains demo model automatically
- Requires **no private artifacts**

---

# 🧠 Tech Stack

| Layer | Technology |
|--------|------------|
| Model | LightGBM |
| API | FastAPI |
| Dashboard | Streamlit |
| Preprocessing | scikit-learn |
| Data | pandas |
| Config | YAML |
| Logging | Python logging |

---

# 🏗️ Project Structure

```
FraudShieldAI/
│
├── api/
│   └── app.py                # FastAPI inference service
│
├── src/
│   ├── data.py               # Data loading utilities
│   ├── features.py           # Preprocessing builder
│   └── train.py              # Training + threshold tuning
│
├── scripts/
│   └── make_demo_data.py     # Synthetic dataset generator
│
├── config/
│   ├── training.yaml
│   └── training_demo.yaml
│
├── data/demo/                # Auto-generated demo dataset
├── models/demo/              # Demo model artifact
│
├── app_streamlit.py          # Optional dashboard
├── requirements.txt
└── README.md
```

---

# 🧩 Installation

```
git clone https://github.com/georgekndh/FraudShieldAI.git
cd FraudShieldAI

python -m venv .venv
.\.venv\Scripts\activate      # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

---

# 🧪 Demo Mode (Zero Setup Required)

This mode runs the full system without any private datasets.

## 1️⃣ Generate Demo Dataset

```
python -m scripts.make_demo_data
```

## 2️⃣ Train Demo Model

```
python -m src.train \
  --config config/training_demo.yaml \
  --input data/demo/transactions_demo.parquet \
  --out models/demo/fraudshield_demo.pkl
```

## 3️⃣ Start API in Demo Mode

Windows
```
set DEMO_MODE=1
uvicorn api.app:app --reload
```

macOS/Linux
```
export DEMO_MODE=1
uvicorn api.app:app --reload
```

Open:
```
http://localhost:8000/docs
```

---

# 🏭 Production Mode

Train using real dataset:

```
python -m src.train \
  --config config/training.yaml \
  --input data/raw/transactions.parquet \
  --out models/fraudshield_lgbm.pkl
```

Run API:

```
set DEMO_MODE=0
set MODEL_PATH=models/fraudshield_lgbm.pkl
uvicorn api.app:app --reload
```

---

# 🌐 API Endpoints

### GET /health

Returns model status:

```
{
  "status": "ok",
  "demo_mode": true,
  "model_loaded": true,
  "threshold": 0.34
}
```

---

### POST /score

```
{
  "data": {
    "Time": 12345,
    "Amount": 78.50,
    "V1": -1.23,
    ...
    "V28": -0.42
  }
}
```

Response:

```
{
  "fraud_probability": 0.9123,
  "flag": 1,
  "threshold": 0.34
}
```

---

### GET /model-info

Returns top-K feature importances.

---

### POST /reload-model

Reloads model bundle without server restart.

---

# 📊 Streamlit Dashboard (Optional)

```
streamlit run app_streamlit.py
```

Includes:
- Probability histograms
- Threshold slider
- SHAP explainability plots
- Transaction inspection

---

# 🔐 Environment Variables

| Variable | Default | Purpose |
|-----------|----------|----------|
| DEMO_MODE | 0 | Enable demo auto-bootstrap |
| MODEL_PATH | models/fraudshield_lgbm.pkl | Model path |
| THRESHOLD | 0.55 | Override classification threshold |
| PRED_LOG | models/predictions_log.csv | Prediction audit log |

---

# 🎯 Design Philosophy

FraudShield AI follows production-oriented ML principles:

- Environment-aware configuration  
- Deterministic artifact loading  
- Strict input schema validation  
- Transparent feature importance  
- No reliance on private committed data  
- Reproducible demo environment  

---

# 🚀 Status

FraudShield AI is ready for:

- Public GitHub hosting  
- Technical portfolio evaluation  
- Demo deployments  
- Extension into production architecture  

Dockerization and monitoring integrations can be added next.