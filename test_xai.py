"""
idss/test_xai.py
-----------------
Tests LIME and SHAP explainability on GBR model
and rule-based explanation on scheduling decisions.
Run from the idss folder:
    C:\Python311\python.exe test_xai.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from models.predictive.gbr_model import engineer_features
from models.xai.xai_explainer import (
    explain_gbr_lime,
    explain_gbr_shap,
    explain_schedule_decision,
)
from adapters.cloud_adapter import CloudAdapter
from models.optimisation.ilp_solver import solve_from_df
from schema import AbstractTask, AbstractResource

print("=" * 55)
print("  IDSS — XAI Explainability Test")
print("=" * 55)

# ── Load data and engineer features ──────────────────────────────
DATA_PATH = os.path.join("data", "processed", "synthetic_all.parquet")
df        = pd.read_parquet(DATA_PATH)
print(f"\n  Loaded {len(df):,} tasks")

features_df = engineer_features(df)
FEATURE_COLS = [
    "task_count_lag1", "task_count_lag2", "task_count_lag3",
    "mean_duration_lag1", "mean_duration_lag2", "mean_duration_lag3",
    "mean_cpu_lag1", "mean_cpu_lag2", "mean_cpu_lag3",
    "task_count_roll3", "mean_duration_roll3",
    "mean_cpu", "mean_mem", "fail_rate", "time_of_day",
]
features_df = features_df.dropna(subset=FEATURE_COLS)
X           = features_df[FEATURE_COLS].values

print(f"  Feature windows available: {len(X)}")

# ── LIME explanation — arrival model ─────────────────────────────
print("\n  [Test 1] LIME explanation — arrival rate model")
lime_result = explain_gbr_lime(
    X_train    = X,
    x_instance = X[-1],
    model_type = "arrival",
)
print(f"  Predicted arrival rate : {lime_result['prediction']} tasks/window")
print(f"  LIME fidelity (R²)     : {lime_result['fidelity_r2']}")
print(f"  Top 5 influential features:")
for f in lime_result["top_features"]:
    bar = "+" if f["weight"] > 0 else "-"
    print(f"    [{bar}] {f['feature']:<30} weight: {f['weight']}")

# ── LIME explanation — duration model ────────────────────────────
print("\n  [Test 2] LIME explanation — duration model")
lime_dur = explain_gbr_lime(
    X_train    = X,
    x_instance = X[-1],
    model_type = "duration",
)
print(f"  Predicted mean duration : {lime_dur['prediction']} seconds")
print(f"  LIME fidelity (R²)      : {lime_dur['fidelity_r2']}")
print(f"  Top 5 influential features:")
for f in lime_dur["top_features"]:
    bar = "+" if f["weight"] > 0 else "-"
    print(f"    [{bar}] {f['feature']:<30} weight: {f['weight']}")

# ── SHAP explanation — arrival model ─────────────────────────────
print("\n  [Test 3] SHAP explanation — arrival rate model")
shap_result = explain_gbr_shap(
    X_train   = X,
    X_explain = X[-5:],
    model_type = "arrival",
)
print(f"  Base value (expected prediction): {shap_result['base_value']}")
print(f"  Top 5 features by SHAP value:")
for f in shap_result["top_features"]:
    print(f"    {f['feature']:<30} SHAP: {f['shap_value']}")

# ── SHAP explanation — duration model ────────────────────────────
print("\n  [Test 4] SHAP explanation — duration model")
shap_dur = explain_gbr_shap(
    X_train    = X,
    X_explain  = X[-5:],
    model_type = "duration",
)
print(f"  Base value (expected prediction): {shap_dur['base_value']}")
print(f"  Top 5 features by SHAP value:")
for f in shap_dur["top_features"]:
    print(f"    {f['feature']:<30} SHAP: {f['shap_value']}")

# ── Rule-based schedule explanation ──────────────────────────────
print("\n  [Test 5] Rule-based explanation for a scheduling decision")

adapter = CloudAdapter()
df_low  = pd.read_parquet(os.path.join("data", "processed",
                                        "synthetic_low_load.parquet"))
result  = solve_from_df(df_low, adapter, n_tasks=10, n_resources=3)

sample      = df_low.sample(n=10, random_state=42)
tasks       = [adapter.map_task(row) for row in sample.to_dict("records")]
machine_ids = sample["machine_id"].unique()[:3]
resources   = [adapter.map_resource({"machine_id": mid})
               for mid in machine_ids]

if result["schedule"]:
    entry    = result["schedule"][0]
    task     = tasks[0]
    resource = resources[0]

    explanation = explain_schedule_decision(task, resource, entry)

    print(f"\n  Task     : {explanation['task_id']}")
    print(f"  Resource : {explanation['resource_id']}")
    print(f"  SLA met  : {explanation['sla_met']}")
    print(f"  Reasoning:")
    for i, reason in enumerate(explanation["reasons"], 1):
        print(f"    {i}. {reason}")

print("\n" + "=" * 55)
print("  XAI tests complete.")
print("=" * 55)