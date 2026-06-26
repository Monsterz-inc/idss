"""
idss/data/preprocess_alibaba.py
---------------------------------
Preprocesses the Alibaba Cluster Trace v2018 into a clean Parquet file.

BEFORE RUNNING:
  1. Download batch_task.tar.gz from:
     http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2018Traces/batch_task.tar.gz
  2. Extract and place batch_task.csv inside: idss/data/raw/alibaba/
  3. Run: C:\Python311\python.exe data\preprocess_alibaba.py
"""
import pandas as pd
import numpy as np
import os, glob

SAMPLE_SIZE = 50_000
RANDOM_SEED = 42

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw", "alibaba")
OUT_DIR = os.path.join(os.path.dirname(__file__), "processed")


def preprocess():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Find file
    patterns = [
        os.path.join(RAW_DIR, "batch_task.csv"),
        os.path.join(RAW_DIR, "*.csv"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {RAW_DIR}.\n"
            "Download from: https://github.com/alibaba/clusterdata\n"
            "Place the CSV in data/raw/alibaba/"
        )

    print(f"  Loading: {files[0]}")

    # Alibaba batch_task.csv has no header row
    # Official columns per schema.txt:
    # task_name, instance_num, job_name, task_type, status,
    # start_time, end_time, plan_cpu, plan_mem
    df = pd.read_csv(
        files[0],
        low_memory = False,
        header     = None,
        names      = [
            "task_name", "instance_num", "job_name",
            "task_type", "status", "start_time",
            "end_time",  "plan_cpu",    "plan_mem"
        ]
    )
    print(f"  Raw shape: {df.shape}")

    # ── Rename to standard names ──────────────────────────────────
    df = df.rename(columns={
        "task_name": "task_id",
        "job_name":  "job_id",
        "plan_cpu":  "cpu_request",
        "plan_mem":  "mem_request",
    })

    # ── Drop rows with nulls in critical columns ──────────────────
    critical = ["start_time", "end_time", "cpu_request", "mem_request"]
    df = df.dropna(subset=critical)

    # ── Convert to numeric ────────────────────────────────────────
    for col in ["start_time", "end_time", "cpu_request", "mem_request"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=critical)

    # ── Remove invalid values (-1 and 101 per Alibaba docs) ───────
    df = df[df["cpu_request"] >= 0]
    df = df[df["mem_request"] >= 0]
    df = df[df["cpu_request"] <= 100]
    df = df[df["mem_request"] <= 100]

    # ── Normalise CPU and memory to [0, 1] ────────────────────────
    # Alibaba scales these between 0-100
    df["cpu_request"] = (df["cpu_request"] / 100.0).clip(0.0, 1.0)
    df["mem_request"] = (df["mem_request"] / 100.0).clip(0.0, 1.0)

    # ── Compute duration ──────────────────────────────────────────
    df["duration"] = (df["end_time"] - df["start_time"]).clip(lower=1.0)

    # ── Normalise timestamps to start from 0 ─────────────────────
    t_min = df["start_time"].min()
    df["start_time"] = df["start_time"] - t_min
    df["end_time"]   = df["end_time"]   - t_min

    # ── Fill missing status ───────────────────────────────────────
    if "status" not in df.columns:
        df["status"] = "Success"
    df["status"] = df["status"].fillna("Success")

    # ── Add derived columns ───────────────────────────────────────
    df["machine_id"] = df.get("machine_id", pd.Series(
        np.random.randint(0, 20, size=len(df)),
        index=df.index
    ))
    df["job_type"] = "batch"
    df["scenario"] = "alibaba"

    print(f"  After cleaning: {len(df):,} rows")

    # ── Stratified sample ─────────────────────────────────────────
    if len(df) > SAMPLE_SIZE:
        df = df.sample(
            n            = SAMPLE_SIZE,
            random_state = RANDOM_SEED
        ).reset_index(drop=True)
        print(f"  Sampled to: {SAMPLE_SIZE:,} rows")

    # ── Save ──────────────────────────────────────────────────────
    out_path = os.path.join(OUT_DIR, "alibaba_processed.parquet")
    df.to_parquet(out_path, index=False)
    print(f"  Saved to: {out_path}")

    # ── Summary ───────────────────────────────────────────────────
    print("\n  Column summary:")
    summary_cols = [c for c in
                    ["duration", "cpu_request", "mem_request"]
                    if c in df.columns]
    print(df[summary_cols].describe().round(3))
    print(f"\n  Status distribution:")
    print(df["status"].value_counts())

    return df


if __name__ == "__main__":
    print("=== Preprocessing Alibaba Cluster Trace v2018 ===")
    preprocess()
    print("\nDone.")