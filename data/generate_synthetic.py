"""
idss/data/generate_synthetic.py
--------------------------------
Generates synthetic workloads calibrated to Alibaba trace statistics.
Run this first to get data immediately without needing the real trace.

Usage:
    python data/generate_synthetic.py
"""
import numpy as np
import pandas as pd
import os

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

SCENARIOS = {
    "low_load":     {"n": 10000, "lam": 5,  "failure_rate": 0.01},
    "high_load":    {"n": 10000, "lam": 25, "failure_rate": 0.02},
    "bursty":       {"n": 10000, "lam": 15, "failure_rate": 0.02},
    "high_failure": {"n": 10000, "lam": 10, "failure_rate": 0.08},
    "mixed":        {"n": 10000, "lam": 20, "failure_rate": 0.03},
}

MEAN_DURATION = 45.0
MEAN_CPU      = 0.18
MEAN_MEM      = 0.12
N_MACHINES    = 20


def generate_scenario(name, n, lam, failure_rate):
    inter_arrivals = np.random.exponential(1.0 / lam, size=n)
    arrival_times  = np.cumsum(inter_arrivals)

    durations    = np.random.exponential(MEAN_DURATION, size=n).clip(1.0, 600.0)
    cpu_requests = np.random.exponential(MEAN_CPU, size=n).clip(0.05, 0.95)
    mem_requests = np.random.exponential(MEAN_MEM, size=n).clip(0.05, 0.95)
    priorities   = np.random.choice([1, 2, 3, 4, 5], size=n, p=[0.1, 0.3, 0.3, 0.2, 0.1])
    machine_ids  = np.random.randint(0, N_MACHINES, size=n)
    statuses     = np.where(np.random.rand(n) < failure_rate, "Failed", "Success")

    if name == "bursty":
        burst_indices = np.random.choice(n, size=300, replace=False)
        arrival_times[burst_indices] = arrival_times[burst_indices] * 0.05

    df = pd.DataFrame({
        "task_id":     [f"{name}_{i:06d}" for i in range(n)],
        "start_time":  arrival_times,
        "end_time":    arrival_times + durations,
        "duration":    durations,
        "cpu_request": cpu_requests,
        "mem_request": mem_requests,
        "cpu_usage":   (cpu_requests * np.random.uniform(0.6, 1.0, n)).clip(0, 1),
        "mem_usage":   (mem_requests * np.random.uniform(0.6, 1.0, n)).clip(0, 1),
        "priority":    priorities,
        "machine_id":  machine_ids,
        "status":      statuses,
        "scenario":    name,
        "job_type":    np.random.choice(["batch", "online"], size=n, p=[0.7, 0.3]),
    })

    return df.sort_values("start_time").reset_index(drop=True)


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "processed")
    os.makedirs(out_dir, exist_ok=True)

    all_frames = []
    for name, params in SCENARIOS.items():
        print(f"  Generating: {name} ({params['n']:,} tasks) ...", end=" ")
        df = generate_scenario(name, **params)
        path = os.path.join(out_dir, f"synthetic_{name}.parquet")
        df.to_parquet(path, index=False)
        all_frames.append(df)
        print(f"saved.")

    combined = pd.concat(all_frames, ignore_index=True)
    combined_path = os.path.join(out_dir, "synthetic_all.parquet")
    combined.to_parquet(combined_path, index=False)
    print(f"\n  Combined: {len(combined):,} tasks saved to {combined_path}")
    print("\n  Summary:")
    print(combined[["duration","cpu_request","mem_request","priority"]].describe().round(3))


if __name__ == "__main__":
    print("=== Generating synthetic workloads ===")
    main()
    print("\nDone. Data is ready in data/processed/")