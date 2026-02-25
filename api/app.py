"""
FraudShield AI - FastAPI inference service (Kaggle creditcard schema)

- Loads model bundle: {"pipeline": sklearn.Pipeline, "threshold": float}
- Scores single and batch requests
- Logs request latency + per-prediction audit log (CSV)
"""

from __future__ import annotations

import csv
import logging
import os
import time
import uuid
import numpy as np
from typing import Any, Dict, List, Optional, Union

import joblib
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import subprocess
from datetime import datetime


# ------------------
# Logging
# ------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("fraudshield")

DEMO_MODE = os.getenv("DEMO_MODE", "0") == "1"

# Demo defaults (used only if DEMO_MODE=1)
DEMO_MODEL_PATH = os.getenv("DEMO_MODEL_PATH", "models/demo/fraudshield_demo.pkl")
DEMO_DATA_PATH = os.getenv("DEMO_DATA_PATH", "data/demo/transactions_demo.parquet")
DEMO_TRAIN_CFG = os.getenv("DEMO_TRAIN_CFG", "config/training_demo.yaml")
# ------------------
# Config
# ------------------
load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH", DEMO_MODEL_PATH if DEMO_MODE else "models/fraudshield_lgbm.pkl")
DEFAULT_THRESHOLD = float(os.getenv("THRESHOLD", "0.55"))
PRED_LOG = os.getenv("PRED_LOG", "models/predictions_log.csv")

# Kaggle creditcard dataset feature set
REQUIRED_COLS: List[str] = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]




# optional version metadata
APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
GIT_SHA = os.getenv("GIT_SHA", "dev")

# ------------------
# Model holder
# ------------------
class ModelState:
    def __init__(self) -> None:
        self.pipe = None
        self.threshold = DEFAULT_THRESHOLD
        self.model_path = MODEL_PATH

    def load(self) -> None:
        """Load bundle from disk (idempotent)."""
        if not os.path.exists(self.model_path):
            self.pipe = None
            self.threshold = DEFAULT_THRESHOLD
            log.warning(f"Model bundle not found at: {self.model_path}")
            return

        bundle = joblib.load(self.model_path)
        pipe = bundle.get("pipeline")
        thr = bundle.get("threshold", self.threshold)

        if pipe is None:
            raise RuntimeError("Model bundle missing key: 'pipeline'")

        self.pipe = pipe
        self.threshold = float(thr)
        log.info(f"Loaded model bundle from {self.model_path} (threshold={self.threshold:.4f})")

def ensure_demo_artifacts() -> None:
    """
    If DEMO_MODE is enabled and demo model doesn't exist, generate demo data + train demo model.
    This makes the repo runnable right after clone.
    """
    if not DEMO_MODE:
        return

    if os.path.exists(MODEL_PATH):
        return

    log.warning("[demo] Demo model not found, bootstrapping demo artifacts...")

    # Ensure directories
    _ensure_parent_dir(DEMO_MODEL_PATH)
    _ensure_parent_dir(DEMO_DATA_PATH)

    # 1) Generate demo data (script you will add to repo)
    # python scripts/make_demo_data.py --out data/demo/transactions_demo.parquet ...
    try:
        subprocess.check_call(
            ["python", "scripts/make_demo_data.py", "--out", DEMO_DATA_PATH, "--n", "5000", "--seed", "42"]
        )
    except Exception as e:
        raise RuntimeError(f"[demo] Failed generating demo data: {e}")

    # 2) Train demo model (train.py should accept these args)
    # python train.py --config config/training_demo.yaml --input data/demo/transactions_demo.parquet --out models/demo/fraudshield_demo.pkl
    try:
        subprocess.check_call(
            ["python", "train.py", "--config", DEMO_TRAIN_CFG, "--input", DEMO_DATA_PATH, "--out", DEMO_MODEL_PATH]
        )
    except Exception as e:
        raise RuntimeError(f"[demo] Failed training demo model: {e}")

    log.info(f"[demo] Bootstrapped demo model at {DEMO_MODEL_PATH}")

STATE = ModelState()

STATE.load()


# ------------------
# Pydantic schemas
# ------------------
class TxnPayload(BaseModel):
    data: Dict[str, Any] = Field(..., description="Transaction features")


class BatchPayload(BaseModel):
    data: List[Dict[str, Any]] = Field(..., description="List of transaction feature dicts")


# ------------------
# App init
# ------------------
app = FastAPI(title="FraudShield AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------
# Helpers
# ------------------


def ensure_model_loaded() -> None:
    if STATE.pipe is None:
        raise HTTPException(
            status_code=500,
            detail=f"Model not loaded. Expected bundle at '{STATE.model_path}'. Train and save the model first.",
        )


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def validate_row(row: Dict[str, Any]) -> None:
    missing = [c for c in REQUIRED_COLS if c not in row]
    if missing:
        raise HTTPException(status_code=400, detail={"missing_features": missing})


def coerce_numeric_row(row: Dict[str, Any]) -> Dict[str, float]:
    """
    Convert required feature values to float.
    Raises 400 if conversion fails.
    """
    out: Dict[str, float] = {}
    bad: Dict[str, Any] = {}
    for c in REQUIRED_COLS:
        try:
            out[c] = float(row[c])
        except Exception:
            bad[c] = row.get(c)
    if bad:
        raise HTTPException(status_code=400, detail={"non_numeric_features": bad})
    return out


def log_prediction(features: Dict[str, float], proba: float, flag_int: int) -> None:
    _ensure_parent_dir(PRED_LOG)
    exists = os.path.exists(PRED_LOG)

    with open(PRED_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ts", "proba", "flag", *REQUIRED_COLS])
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "ts": int(time.time()),
                "proba": float(proba),
                "flag": int(flag_int),
                **{k: features.get(k) for k in REQUIRED_COLS},
            }
        )


def predict_one(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Score a single transaction dict.
    """
    ensure_model_loaded()
    validate_row(row)
    row_num = coerce_numeric_row(row)

    df = pd.DataFrame([row_num], columns=REQUIRED_COLS)
    proba = float(STATE.pipe.predict_proba(df)[0, 1])
    flag_int = int(proba >= STATE.threshold)

    # audit log
    log_prediction(row_num, proba, flag_int)

    # app log
    log.info(f"[score] prob={proba:.6g} flag={flag_int} threshold={STATE.threshold:.4f}")

    return {"fraud_probability": proba, "flag": flag_int, "threshold": STATE.threshold}


def predict_many(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Score a batch of transactions.
    """
    ensure_model_loaded()
    if not rows:
        raise HTTPException(status_code=400, detail="Empty batch.")

    rows_num: List[Dict[str, float]] = []
    for r in rows:
        validate_row(r)
        rows_num.append(coerce_numeric_row(r))

    df = pd.DataFrame(rows_num, columns=REQUIRED_COLS)
    probs = STATE.pipe.predict_proba(df)[:, 1]
    flags = (probs >= STATE.threshold).astype(int)

    out: List[Dict[str, Any]] = []
    for r, p, fl in zip(rows_num, probs, flags):
        p = float(p)
        fl = int(fl)
        log_prediction(r, p, fl)
        out.append({"fraud_probability": p, "flag": fl, "threshold": STATE.threshold})

    log.info(f"[score-batch] n={len(out)} threshold={STATE.threshold:.4f}")
    return out

def get_model_feature_importance():
    ensure_model_loaded()

    # Pipeline expected: ("pre", preprocessor), ("clf", LGBMClassifier)
    pre = getattr(STATE.pipe, "named_steps", {}).get("pre") if hasattr(STATE.pipe, "named_steps") else None
    clf = getattr(STATE.pipe, "named_steps", {}).get("clf") if hasattr(STATE.pipe, "named_steps") else None

    if clf is None:
        # fallback: maybe the pipe itself is the estimator
        clf = STATE.pipe

    if not hasattr(clf, "feature_importances_"):
        raise HTTPException(status_code=500, detail="Model has no feature_importances_")

    importances = clf.feature_importances_
    importances = np.asarray(importances, dtype=float)

    # Best-effort feature names
    feat_names = None
    if pre is not None:
        try:
            feat_names = list(pre.get_feature_names_out())
        except Exception:
            feat_names = None

    if feat_names is None:
        # fallback: if no preprocessor names, use REQUIRED_COLS
        feat_names = REQUIRED_COLS

    # Align sizes defensively
    n = min(len(feat_names), len(importances))
    rows = [{"feature": feat_names[i], "importance": float(importances[i])} for i in range(n)]
    rows.sort(key=lambda x: x["importance"], reverse=True)

    return rows



# ------------------
# Middleware (request logs)
# ------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    rid = str(uuid.uuid4())[:8]
    start = time.time()
    try:
        response = await call_next(request)
        ms = (time.time() - start) * 1000
        log.info(f"[{rid}] {request.method} {request.url.path} -> {response.status_code} ({ms:.1f}ms)")
        return response
    except Exception as e:
        ms = (time.time() - start) * 1000
        log.exception(f"[{rid}] {request.method} {request.url.path} -> ERROR ({ms:.1f}ms): {e}")
        raise


# ------------------
# Endpoints
# ------------------
@app.get("/")
def root():
    return {"message": "FraudShield AI is running", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": STATE.pipe is not None,
        "model_path": STATE.model_path,
        "threshold": STATE.threshold,
    }


@app.get("/schema")
def schema():
    return {"required_features": REQUIRED_COLS}


@app.get("/model-info")
def model_info(top_k: int = 30):
    rows = get_model_feature_importance()[:top_k]
    return {
        "model_loaded": STATE.pipe is not None,
        "threshold": STATE.threshold,
        "top_k": top_k,
        "feature_importance": rows,
    }

# added endpoint for demo 
@app.get("/version")
def version():
    return {
        "app_version": APP_VERSION,
        "git_sha": GIT_SHA,
        "demo_mode": DEMO_MODE,
        "model_path": STATE.model_path,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
    }



@app.post("/score")
def score(payload: TxnPayload):
    return predict_one(payload.data)


# Supports BOTH formats:
# 1) {"data": [ {..}, {..} ]}   (recommended)
# 2) [ {"data": {...}}, {"data": {...}} ]  (legacy)
@app.post("/score-batch")
def score_batch(payload: Union[BatchPayload, List[TxnPayload]]):
    if isinstance(payload, list):
        rows = [p.data for p in payload]
        return predict_many(rows)
    return predict_many(payload.data)


@app.post("/reload-model")
def reload_model():
    """
    Reload the model bundle without restarting the server.
    """
    try:
        STATE.load()
        ensure_model_loaded()
        return {"ok": True, "model_loaded": True, "threshold": STATE.threshold, "model_path": STATE.model_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reload failed: {e}")
