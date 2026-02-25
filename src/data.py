import pandas as pd
from typing import Iterable

def read_table(path: str) -> pd.DataFrame:
  return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)

def cast_types(df: pd.DataFrame, dt_cols: Iterable[str] = ("event_time",)) -> pd.DataFrame:
  for c in dt_cols:
    if c in df.columns:
      df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
  return df
