"""
idss/test_gbr.py
-----------------
Tests the GBR predictive model against synthetic data.
Run from the idss folder:
    python test_gbr.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from models.predictive.gbr_model import train, predict

print("=" * 55)
print("  IDSS — GBR Predictive Model Test")
print("=" * 55)

# ── Load synthetic data ───────────────────────────────────────────
DATA_PATH = os.path.join("data", "processed", "synthetic_all.parquet")
df = pd.read_parquet(DATA_PATH)
print(f"\n  Loaded {len(df):,} tasks from synthetic_all")

# ── Train ─────────────────────────────────────────────────────────
print("\n  Training models ...")
results = train(df)

print(f"\n  Windows used for training : {results['n_windows']}")
print(f"\n  Arrival Rate Model:")
print(f"    MAE : {results['arrival_metrics']['MAE']} tasks/window")
print(f"    R²  : {results['arrival_metrics']['R2']}")
print(f"\n  Duration Model:")
print(f"    MAE : {results['duration_metrics']['MAE']} seconds")
print(f"    R²  : {results['duration_metrics']['R2']}")

# ── Predict on recent slice ───────────────────────────────────────
print("\n  Running prediction on last 500 tasks ...")
recent = df.tail(5000)
forecast = predict(recent)

print(f"\n  Forecast for next 5-minute window:")
print(f"    Expected task arrivals : {forecast['arrival_rate']}")
print(f"    Expected mean duration : {forecast['mean_duration']} seconds")

# ── Test across all 5 scenarios ───────────────────────────────────
print("\n  Forecasts per scenario:")
print(f"  {'Scenario':<20} {'Arrivals':>10} {'Duration':>10}")
print("  " + "-" * 42)

scenarios = ["low_load", "high_load", "bursty", "high_failure", "mixed"]
for sc in scenarios:
    path = os.path.join("data", "processed", f"synthetic_{sc}.parquet")
    df_sc = pd.read_parquet(path)
    fc = predict(df_sc.tail(3000))
    print(f"  {sc:<20} {fc['arrival_rate']:>10} {fc['mean_duration']:>10}")

print("\n" + "=" * 55)
print("  GBR Predictive Model tests complete.")
print("=" * 55)