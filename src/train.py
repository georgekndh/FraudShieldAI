import argparse, joblib, yaml
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from lightgbm import LGBMClassifier
from src.data import read_table, cast_types
from src.features import build_preprocessor

def split_time(df, dt_col, train_end, valid_end, target):
  tr = df[df[dt_col] <= train_end]
  va = df[(df[dt_col] > train_end) & (df[dt_col] <= valid_end)]
  return tr.drop(columns=[target]), tr[target], va.drop(columns=[target]), va[target]

def main(cfg_path, input_path, model_out):
  cfg = yaml.safe_load(open(cfg_path))
  df = cast_types(read_table(input_path), [cfg["datetime_col"]])
  pre = build_preprocessor(df, cfg["target"])
  clf = LGBMClassifier(**cfg["lgbm_params"], class_weight=cfg.get("class_weight"))
  pipe = Pipeline([("pre", pre), ("clf", clf)])

  X_tr, y_tr, X_va, y_va = split_time(df, cfg["datetime_col"], cfg["split"]["train_end"], cfg["split"]["valid_end"], cfg["target"])
  pipe.fit(X_tr, y_tr)
  p = pipe.predict_proba(X_va)[:,1]
  auc = roc_auc_score(y_va, p)
  best_t, best_f1 = 0.5, 0.0
  for t in [i/100 for i in range(5,95)]:
    pred = (p >= t).astype(int)
    _,_,f1,_ = precision_recall_fscore_support(y_va, pred, average="binary", zero_division=0)
    if f1 > best_f1: best_f1, best_t = f1, t
  print(f"Valid AUC: {auc:.4f} | Best F1: {best_f1:.4f} @ {best_t:.2f}")
  joblib.dump({"pipeline": pipe, "threshold": best_t}, model_out)

if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--config", default="config/training.yaml")
  ap.add_argument("--input", default="data/raw/transactions.parquet")
  ap.add_argument("--out", default="models/fraudshield_lgbm.pkl")
  a = ap.parse_args()
  main(a.config, a.input, a.out)
