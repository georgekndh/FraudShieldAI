import argparse
import os
import math
import joblib
import yaml
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from lightgbm import LGBMClassifier

from src.data import read_table, cast_types
from src.features import build_preprocessor


def is_missing(x) -> bool:
    """True if x is None / NaN / empty string."""
    if x is None:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    if isinstance(x, str) and x.strip() == "":
        return True
    return False


def split_time(df: pd.DataFrame, dt_col: str, train_end, valid_end, target: str):
    """Time-based split using dt_col thresholds."""
    tr = df[df[dt_col] <= train_end]
    va = df[(df[dt_col] > train_end) & (df[dt_col] <= valid_end)]

    if tr.empty or va.empty:
        raise ValueError(
            f"Time split produced empty set(s). "
            f"train={len(tr)}, valid={len(va)}. "
            f"Check split bounds or dt_col distribution."
        )

    return tr.drop(columns=[target]), tr[target], va.drop(columns=[target]), va[target]


def split_random(df: pd.DataFrame, target: str, test_size=0.2, seed=42):
    """Stratified random split (recommended for Kaggle creditcard.csv)."""
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def main(cfg_path: str, input_path: str, model_out: str):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    target = cfg.get("target", "is_fraud")
    dt_col = cfg.get("datetime_col", None)

    df = read_table(input_path)

    # Only cast datetime types if dt_col is provided AND exists in df
    if not is_missing(dt_col) and dt_col in df.columns:
        df = cast_types(df, [dt_col])

    if target not in df.columns:
        raise KeyError(
            f"Target column '{target}' not found. "
            f"Available columns: {list(df.columns)[:10]} ... (total {len(df.columns)})"
        )

    pre = build_preprocessor(df, target)
    clf = LGBMClassifier(**cfg["lgbm_params"], class_weight=cfg.get("class_weight"))
    pipe = Pipeline([("pre", pre), ("clf", clf)])

    # Choose split strategy
    use_time_split = (
        not is_missing(dt_col)
        and dt_col in df.columns
        and "split" in cfg
        and "train_end" in cfg["split"]
        and "valid_end" in cfg["split"]
    )

    if use_time_split:
        X_tr, y_tr, X_va, y_va = split_time(
            df,
            dt_col,
            cfg["split"]["train_end"],
            cfg["split"]["valid_end"],
            target,
        )
        split_used = f"time split on '{dt_col}'"
    else:
        test_size = cfg.get("random_split", {}).get("test_size", 0.2)
        seed = cfg.get("random_split", {}).get("seed", 42)
        X_tr, X_va, y_tr, y_va = split_random(df, target, test_size=test_size, seed=seed)
        split_used = f"random stratified split (test_size={test_size})"

    print(f"Split: {split_used}")
    print(f"Train rows: {len(X_tr):,} | Valid rows: {len(X_va):,}")

    pipe.fit(X_tr, y_tr)

    p = pipe.predict_proba(X_va)[:, 1]
    auc = roc_auc_score(y_va, p)

    best_t, best_f1 = 0.5, 0.0
    for t in [i / 100 for i in range(5, 95)]:
        pred = (p >= t).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            y_va, pred, average="binary", zero_division=0
        )
        if f1 > best_f1:
            best_f1, best_t = f1, t

    print(f"Valid AUC: {auc:.4f} | Best F1: {best_f1:.4f} @ {best_t:.2f}")

    os.makedirs(os.path.dirname(model_out) or ".", exist_ok=True)
    joblib.dump({"pipeline": pipe, "threshold": best_t}, model_out)
    print(f"Saved model bundle -> {model_out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/training.yaml")
    ap.add_argument("--input", default="data/raw/transactions.parquet")
    ap.add_argument("--out", default="models/fraudshield_lgbm.pkl")
    args = ap.parse_args()
    main(args.config, args.input, args.out)
