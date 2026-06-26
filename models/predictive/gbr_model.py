"""
idss/models/predictive/gbr_model.py
-------------------------------------
Gradient Boosting Regressor for workload forecasting.
Predicts: (1) task arrival rate, (2) mean task duration
over the next 30-second window.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


MODEL_DIR           = os.path.join(os.path.dirname(__file__))
ARRIVAL_MODEL_PATH  = os.path.join(MODEL_DIR, "gbr_arrival.pkl")
DURATION_MODEL_PATH = os.path.join(MODEL_DIR, "gbr_duration.pkl")
SCALER_PATH         = os.path.join(MODEL_DIR, "gbr_scaler.pkl")

WINDOW_SIZE = 30    # 30-second window in seconds


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer time-series features from the raw task DataFrame.
    Each row represents one 30-second window.
    """
    df = df.sort_values("start_time").copy()

    # Bin tasks into windows
    df["window"] = (df["start_time"] // WINDOW_SIZE).astype(int)

    # Aggregate per window
    agg = df.groupby("window").agg(
        task_count    = ("task_id",     "count"),
        mean_duration = ("duration",    "mean"),
        mean_cpu      = ("cpu_request", "mean"),
        mean_mem      = ("mem_request", "mean"),
        fail_rate     = ("status",      lambda x: (x == "Failed").mean()),
    ).reset_index()

    # Lag features (previous 1, 2, 3 windows)
    for lag in [1, 2, 3]:
        agg[f"task_count_lag{lag}"]    = agg["task_count"].shift(lag)
        agg[f"mean_duration_lag{lag}"] = agg["mean_duration"].shift(lag)
        agg[f"mean_cpu_lag{lag}"]      = agg["mean_cpu"].shift(lag)

    # Rolling mean features
    agg["task_count_roll3"]    = agg["task_count"].rolling(3).mean()
    agg["mean_duration_roll3"] = agg["mean_duration"].rolling(3).mean()

    # Time-of-day proxy
    agg["time_of_day"] = agg["window"] % 12

    # Drop rows with NaN from lag/rolling
    agg = agg.dropna().reset_index(drop=True)

    return agg


def train(df: pd.DataFrame) -> dict:
    """
    Train GBR models for arrival rate and duration forecasting.
    Saves models to disk. Returns evaluation metrics.
    """
    print("  Engineering features ...")
    features_df = engineer_features(df)

    feature_cols = [
        "task_count_lag1", "task_count_lag2", "task_count_lag3",
        "mean_duration_lag1", "mean_duration_lag2", "mean_duration_lag3",
        "mean_cpu_lag1", "mean_cpu_lag2", "mean_cpu_lag3",
        "task_count_roll3", "mean_duration_roll3",
        "mean_cpu", "mean_mem", "fail_rate", "time_of_day",
    ]

    features_df = features_df.dropna(subset=feature_cols).reset_index(drop=True)

    X          = features_df[feature_cols].values
    y_arrival  = features_df["task_count"].values
    y_duration = features_df["mean_duration"].values

    # Scale features
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train / test split
    X_tr, X_te, ya_tr, ya_te, yd_tr, yd_te = train_test_split(
        X_scaled, y_arrival, y_duration,
        test_size=0.2, random_state=42
    )

    # ── Arrival rate model ──────────────────────────────────────────
    print("  Training arrival rate model ...")
    gbr_arrival = GradientBoostingRegressor(
        n_estimators  = 200,
        learning_rate = 0.05,
        max_depth     = 4,
        subsample     = 0.8,
        random_state  = 42,
    )
    gbr_arrival.fit(X_tr, ya_tr)
    ya_pred = gbr_arrival.predict(X_te)
    arrival_metrics = {
        "MAE": round(mean_absolute_error(ya_te, ya_pred), 3),
        "R2":  round(r2_score(ya_te, ya_pred), 3),
    }

    # ── Duration model ──────────────────────────────────────────────
    print("  Training duration model ...")
    gbr_duration = GradientBoostingRegressor(
        n_estimators  = 200,
        learning_rate = 0.05,
        max_depth     = 4,
        subsample     = 0.8,
        random_state  = 42,
    )
    gbr_duration.fit(X_tr, yd_tr)
    yd_pred = gbr_duration.predict(X_te)
    duration_metrics = {
        "MAE": round(mean_absolute_error(yd_te, yd_pred), 3),
        "R2":  round(r2_score(yd_te, yd_pred), 3),
    }

    # ── Save models ─────────────────────────────────────────────────
    with open(ARRIVAL_MODEL_PATH,  "wb") as f: pickle.dump(gbr_arrival,  f)
    with open(DURATION_MODEL_PATH, "wb") as f: pickle.dump(gbr_duration, f)
    with open(SCALER_PATH,         "wb") as f: pickle.dump(scaler,       f)
    print(f"  Models saved to {MODEL_DIR}")

    return {
        "n_windows":        len(features_df),
        "arrival_metrics":  arrival_metrics,
        "duration_metrics": duration_metrics,
        "feature_cols":     feature_cols,
    }


def predict(df_recent: pd.DataFrame) -> dict:
    """
    Predict arrival rate and mean duration for the next window.
    df_recent should contain a recent slice of task data.
    """
    if not os.path.exists(ARRIVAL_MODEL_PATH):
        raise FileNotFoundError("Models not found. Run train() first.")

    with open(ARRIVAL_MODEL_PATH,  "rb") as f: gbr_arrival  = pickle.load(f)
    with open(DURATION_MODEL_PATH, "rb") as f: gbr_duration = pickle.load(f)
    with open(SCALER_PATH,         "rb") as f: scaler       = pickle.load(f)

    features_df = engineer_features(df_recent)

    feature_cols = [
        "task_count_lag1", "task_count_lag2", "task_count_lag3",
        "mean_duration_lag1", "mean_duration_lag2", "mean_duration_lag3",
        "mean_cpu_lag1", "mean_cpu_lag2", "mean_cpu_lag3",
        "task_count_roll3", "mean_duration_roll3",
        "mean_cpu", "mean_mem", "fail_rate", "time_of_day",
    ]

    # Drop any rows with NaN in feature columns
    features_df = features_df.dropna(subset=feature_cols)

    if len(features_df) == 0:
        return {"arrival_rate": 0, "mean_duration": 0}

    X    = scaler.transform(features_df[feature_cols].values)
    last = X[-1].reshape(1, -1)

    arrival  = max(0, gbr_arrival.predict(last)[0])
    duration = max(0, gbr_duration.predict(last)[0])

    return {
        "arrival_rate":  round(arrival, 2),
        "mean_duration": round(duration, 2),
    }