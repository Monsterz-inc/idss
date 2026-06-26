"""
idss/test_ppo.py
-----------------
Tests the PPO DRL agent using the already trained model.
Run from the idss folder:
    C:\Python311\python.exe test_ppo.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from adapters.cloud_adapter import CloudAdapter
from models.drl.ppo_agent import load, run_episode

print("=" * 55)
print("  IDSS — PPO DRL Agent Evaluation")
print("=" * 55)

# ── Load data ─────────────────────────────────────────────────────
df      = pd.read_parquet(os.path.join("data", "processed",
                                        "synthetic_all.parquet"))
adapter = CloudAdapter()

print(f"\n  Loaded {len(df):,} tasks")

# ── Load already trained model ────────────────────────────────────
print("\n  Loading trained model ...")
model = load()
print("  Model loaded successfully.")

# ── Evaluate on low_load ──────────────────────────────────────────
print("\n  Evaluating on low_load scenario ...")
df_low = pd.read_parquet(os.path.join("data", "processed",
                                       "synthetic_low_load.parquet"))
r1 = run_episode(
    model, df_low, adapter,
    n_tasks=50, n_resources=10,
    forecast={"arrival_rate": 23, "mean_duration": 44}
)
print(f"  Tasks completed  : {r1['tasks_done']}")
print(f"  SLA violations   : {r1['sla_violations']}")
print(f"  Total energy     : {r1['total_energy']} kWh")
print(f"  Total reward     : {r1['total_reward']}")

# ── Evaluate on high_load ─────────────────────────────────────────
print("\n  Evaluating on high_load scenario ...")
df_high = pd.read_parquet(os.path.join("data", "processed",
                                        "synthetic_high_load.parquet"))
r2 = run_episode(
    model, df_high, adapter,
    n_tasks=50, n_resources=10,
    forecast={"arrival_rate": 398, "mean_duration": 44}
)
print(f"  Tasks completed  : {r2['tasks_done']}")
print(f"  SLA violations   : {r2['sla_violations']}")
print(f"  Total energy     : {r2['total_energy']} kWh")
print(f"  Total reward     : {r2['total_reward']}")

# ── Evaluate on bursty ────────────────────────────────────────────
print("\n  Evaluating on bursty scenario ...")
df_bursty = pd.read_parquet(os.path.join("data", "processed",
                                          "synthetic_bursty.parquet"))
r3 = run_episode(
    model, df_bursty, adapter,
    n_tasks=50, n_resources=10,
    forecast={"arrival_rate": 249, "mean_duration": 45}
)
print(f"  Tasks completed  : {r3['tasks_done']}")
print(f"  SLA violations   : {r3['sla_violations']}")
print(f"  Total energy     : {r3['total_energy']} kWh")
print(f"  Total reward     : {r3['total_reward']}")

# ── Evaluate on high_failure ──────────────────────────────────────
print("\n  Evaluating on high_failure scenario ...")
df_fail = pd.read_parquet(os.path.join("data", "processed",
                                        "synthetic_high_failure.parquet"))
r4 = run_episode(
    model, df_fail, adapter,
    n_tasks=50, n_resources=10,
    forecast={"arrival_rate": 263, "mean_duration": 47}
)
print(f"  Tasks completed  : {r4['tasks_done']}")
print(f"  SLA violations   : {r4['sla_violations']}")
print(f"  Total energy     : {r4['total_energy']} kWh")
print(f"  Total reward     : {r4['total_reward']}")

# ── Evaluate on mixed ─────────────────────────────────────────────
print("\n  Evaluating on mixed scenario ...")
df_mixed = pd.read_parquet(os.path.join("data", "processed",
                                         "synthetic_mixed.parquet"))
r5 = run_episode(
    model, df_mixed, adapter,
    n_tasks=50, n_resources=10,
    forecast={"arrival_rate": 439, "mean_duration": 44}
)
print(f"  Tasks completed  : {r5['tasks_done']}")
print(f"  SLA violations   : {r5['sla_violations']}")
print(f"  Total energy     : {r5['total_energy']} kWh")
print(f"  Total reward     : {r5['total_reward']}")

# ── Summary table ─────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  SUMMARY")
print("=" * 55)
print(f"  {'Scenario':<20} {'Done':>6} {'SLA Viol':>10} "
      f"{'Energy':>10} {'Reward':>10}")
print("  " + "-" * 55)

rows = [
    ("low_load",     r1),
    ("high_load",    r2),
    ("bursty",       r3),
    ("high_failure", r4),
    ("mixed",        r5),
]
for name, r in rows:
    print(f"  {name:<20} {r['tasks_done']:>6} "
          f"{r['sla_violations']:>10} "
          f"{r['total_energy']:>10.2f} "
          f"{r['total_reward']:>10.2f}")

print("\n" + "=" * 55)
print("  PPO evaluation complete.")
print("=" * 55)