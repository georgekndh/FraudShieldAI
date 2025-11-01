import os
import time
import csv
from typing import Dict, Any, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# ------------------
# Environment & model
# ------------------
load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH", "models/fraudshield_lgbm.pkl")
THRESHOLD = float(os.getenv("THRESHOLD", 0.55))
PRED_LOG = os.getenv("PRED_LOG", "models/predictions_log.csv")

bundle = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
if not bundle:
    pipe = None
else:
    pipe = bundle.get("pipeline")
    THRESHOLD = float(bundle.get("threshold", THRESHOLD))

# ------------------
# App init
# ------------------
app = FastAPI(title="FraudShield AI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# The Kaggle creditcard dataset feature set
REQUIRED_COLS = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]

class TxnPayload(BaseModel):
    data: Dict[str, Any]

# -------------
# Utilities
# -------------

def ensure_model_loaded():
    if pipe is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Train and save the model bundle first.")


def log_prediction(features: Dict[str, Any], proba: float, flag: int) -> None:
    os.makedirs(os.path.dirname(PRED_LOG), exist_ok=True)
    exists = os.path.exists(PRED_LOG)
    with open(PRED_LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ts", "proba", "flag", *REQUIRED_COLS])
        if not exists:
            writer.writeheader()
        writer.writerow({"ts": int(time.time()), "proba": proba, "flag": flag, **{k: features.get(k) for k in REQUIRED_COLS}})


def validate_row(row: Dict[str, Any]):
    missing = [c for c in REQUIRED_COLS if c not in row]
    if missing:
        raise HTTPException(status_code=400, detail={"missing_features": missing})

# -------------
# Endpoints
# -------------

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": pipe is not None, "threshold": THRESHOLD}


@app.get("/schema")
def schema():
    return {"required_features": REQUIRED_COLS}


@app.post("/score")
def score(payload: TxnPayload):
    ensure_model_loaded()
    row = payload.data
    validate_row(row)
    df = pd.DataFrame([row], columns=REQUIRED_COLS)
    proba = float(pipe.predict_proba(df)[0, 1])
    flag = int(proba >= THRESHOLD)
    log_prediction(row, proba, flag)
    return {"fraud_probability": proba, "flag": flag, "threshold": THRESHOLD}


@app.post("/score-batch")
def score_batch(payload: List[TxnPayload]):
    ensure_model_loaded()
    rows = [p.data for p in payload]
    # validate rows & enforce column order
    for r in rows:
        validate_row(r)
    df = pd.DataFrame(rows, columns=REQUIRED_COLS)
    probs = pipe.predict_proba(df)[:, 1]
    flags = (probs >= THRESHOLD).astype(int)

    # log each row
    for r, p, f in zip(rows, probs, flags):
        log_prediction(r, float(p), int(f))

    return [
        {"fraud_probability": float(p), "flag": int(f), "threshold": THRESHOLD}
        for p, f in zip(probs, flags)
    ]
