import os
import sys
import joblib
import pandas as pd

BUNDLE_PATH = r"C:\Projects\FraudShieldAI\models\fraudshield_lgbm.pkl"
INPUT_PATH  = r"C:\Projects\FraudShieldAI\data\to_score.csv"     # put your raw rows here
OUTPUT_PATH = r"C:\Projects\FraudShieldAI\data\scored.csv"

if not os.path.exists(BUNDLE_PATH):
    sys.exit(f"❌ Missing model bundle: {BUNDLE_PATH}")

bundle = joblib.load(BUNDLE_PATH)
pipe    = bundle["pipeline"] if isinstance(bundle, dict) else bundle
expect  = bundle.get("features") if isinstance(bundle, dict) else None  # optional

if not os.path.exists(INPUT_PATH):
    sys.exit(f"❌ Missing input CSV to score: {INPUT_PATH}")

df = pd.read_csv(INPUT_PATH)

# If we know the expected schema, enforce it (clear error if mismatched).
if expect is not None:
    missing = [c for c in expect if c not in df.columns]
    extra   = [c for c in df.columns if c not in expect]
    if missing:
        sys.exit(
            "❌ Input CSV doesn't match the training schema.\n"
            f"Missing columns: {missing}\nExtra columns: {extra}\n"
            f"Expected EXACT columns: {expect}"
        )
    df = df[expect]  # ensure order

# Score
proba = pipe.predict_proba(df)[:, 1]
df_out = df.copy()
df_out["fraud_probability"] = proba

# Optionally add a friendly id
if "txn_id" not in df_out.columns:
    df_out.insert(0, "txn_id", range(1, len(df_out) + 1))

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df_out.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Scored file saved to {OUTPUT_PATH}")
