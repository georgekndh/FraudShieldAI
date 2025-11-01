# 🛡️ FraudShield AI

> Real-time credit card fraud detection system powered by **LightGBM**, **FastAPI**, and **SHAP explainability**.

FraudShield AI is a complete pipeline for detecting fraudulent transactions in real time — from preprocessing and model training to live API inference and interactive visual explainers.  
Built with a focus on speed, transparency, and interpretability.

---

## ⚙️ Features

- 🧩 **End-to-End Pipeline** — data preprocessing, feature engineering, and model training.
- 🚀 **Real-Time Scoring API** — FastAPI backend for instant transaction risk scoring.
- 🎯 **Optimized Thresholding** — automatically finds the best F1 threshold.
- 🔍 **Explainability** — SHAP force and waterfall plots for local + global interpretation.
- 📊 **Streamlit Dashboard** — visualize predictions, model confidence, and top suspicious transactions.

---

## 🧠 Tech Stack

| Component | Technology |
|------------|-------------|
| Model | LightGBM |
| API | FastAPI |
| Dashboard | Streamlit |
| Preprocessing | scikit-learn, pandas |
| Explainability | SHAP |
| Logging & Metrics | MLflow |

---

## 🏗️ Project Structure

```
FraudShieldAI/
│
├── api/                    # FastAPI backend (app.py)
├── src/                    # Core ML pipeline and utilities
│   ├── data.py             # Data loading helpers
│   ├── features.py         # Feature engineering
│   └── train.py            # Model training & threshold tuning
│
├── notebooks/              # Jupyter notebooks for experiments
├── models/                 # Saved trained models (ignored by git)
├── config/                 # YAML config for model parameters
├── app_streamlit.py        # Streamlit dashboard
├── score_transactions.py   # Batch scoring script
├── requirements.txt        # Dependencies
└── .gitignore
```

---

## 🧩 Installation

```bash
# Clone the repository
git clone https://github.com/georgekndh/FraudShieldAI.git
cd FraudShieldAI

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate   # on Windows
# source .venv/bin/activate  # on macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

---

## 🧮 Train the Model

```bash
python src/train.py --config config/training.yaml                     --input data/raw/transactions.parquet                     --out models/fraudshield_lgbm.pkl
```

This will:
- Train a LightGBM model.
- Optimize the decision threshold for best F1 score.
- Save the model bundle in `/models`.

---

## ⚡ Run the API Server

```bash
uvicorn api.app:app --reload
```

Then open: [http://localhost:8000/docs](http://localhost:8000/docs)

### Example request:
```json
POST /score
{
  "data": {
    "Time": 12345,
    "Amount": 78.50,
    "V1": -1.23,
    "V2": 0.56,
    "...": "...",
    "V28": -0.42
  }
}
```

### Example response:
```json
{
  "fraud_probability": 0.9123,
  "flag": 1,
  "threshold": 0.55
}
```

---

## 💻 Streamlit Dashboard

Launch the dashboard to visualize predictions and SHAP insights:

```bash
streamlit run app_streamlit.py
```

Features:
- Probability histograms  
- Interactive threshold slider  
- SHAP force & waterfall plots  
- Explanation of positive/negative contribution values  

---

## 📈 Explainability Example

SHAP outputs interpret model decisions — e.g.,  
- **Positive values** → increase fraud likelihood  
- **Negative values** → reduce fraud likelihood  
- `E[f(x)]` → model’s expected (base) fraud risk

---

## 🧰 Environment Variables

| Variable | Default | Description |
|-----------|----------|-------------|
| `MODEL_PATH` | `models/fraudshield_lgbm.pkl` | Path to saved model |
| `THRESHOLD` | `0.55` | Default fraud threshold |
| `PRED_LOG` | `models/predictions_log.csv` | Path to logged predictions |
