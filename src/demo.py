# src/demo.py
import numpy as np
import pandas as pd

def make_demo_transactions(
    n: int = 5000,
    seed: int = 42,
    fraud_rate: float = 0.02,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Core numeric signals
    amount = rng.lognormal(mean=3.4, sigma=0.7, size=n)  # skewed like real money
    hour = rng.integers(0, 24, size=n)
    device_age_days = rng.integers(0, 1200, size=n)

    # Categorical-ish fields
    channel = rng.choice(["pos", "ecom", "atm", "transfer"], size=n, p=[0.35, 0.35, 0.15, 0.15])
    country = rng.choice(["JO", "DE", "TR", "US", "GB", "AE"], size=n, p=[0.55, 0.10, 0.15, 0.08, 0.06, 0.06])
    merchant_cat = rng.choice(
        ["grocery", "electronics", "fuel", "travel", "luxury", "gaming", "pharmacy"],
        size=n,
        p=[0.22, 0.14, 0.18, 0.08, 0.05, 0.18, 0.15],
    )
    has_chargeback_history = rng.choice([0, 1], size=n, p=[0.92, 0.08])

    # Fraud logic (not “real”, but realistic-ish)
    # Higher fraud odds for: high amount, ecom, odd hours, newer device, travel/luxury, some countries, chargeback history
    logit = (
        -4.0
        + 0.0009 * (amount - np.mean(amount))
        + 0.35 * (channel == "ecom").astype(float)
        + 0.25 * ((hour <= 5) | (hour >= 23)).astype(float)
        - 0.0010 * device_age_days
        + 0.55 * np.isin(merchant_cat, ["travel", "luxury", "electronics"]).astype(float)
        + 0.35 * np.isin(country, ["TR", "US", "GB"]).astype(float)
        + 0.85 * has_chargeback_history.astype(float)
    )

    probs = 1 / (1 + np.exp(-logit))

    # Calibrate to target fraud_rate by scaling logits a bit
    # Simple approach: shift logits to match mean probability.
    # This keeps the “shape” but moves base rate.
    current_rate = probs.mean()
    shift = np.log(fraud_rate / (1 - fraud_rate)) - np.log(current_rate / (1 - current_rate))
    probs = 1 / (1 + np.exp(-(logit + shift)))

    is_fraud = (rng.random(n) < probs).astype(int)

    df = pd.DataFrame(
        {
            "amount": amount,
            "hour": hour,
            "device_age_days": device_age_days,
            "channel": channel,
            "country": country,
            "merchant_cat": merchant_cat,
            "has_chargeback_history": has_chargeback_history,
            "is_fraud": is_fraud,
        }
    )
    return df