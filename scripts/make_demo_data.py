# scripts/make_demo_data.py
import os
import argparse
from src.demo import make_demo_transactions

def main(out_path: str, n: int, seed: int):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df = make_demo_transactions(n=n, seed=seed)
    if out_path.endswith(".parquet"):
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)
    print(f"Saved demo data -> {out_path} ({len(df):,} rows)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/demo/transactions_demo.parquet")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args.out, args.n, args.seed)