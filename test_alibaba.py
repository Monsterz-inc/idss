"""
idss/test_alibaba.py
---------------------
Evaluates the IDSS on real Alibaba Cluster Trace v2018 data.
Run from the idss folder:
    C:\Python311\python.exe test_alibaba.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from adapters.cloud_adapter import CloudAdapter
from models.optimisation.ilp_solver import solve_from_df
from models.predictive.gbr_model import train, predict
from models.drl.ppo_agent import load, run_episode
from kg.kg_validator import KnowledgeGraph

print("=" * 55)
print("  IDSS — Real Alibaba Data Evaluation")
print("=" * 55)

# ── Load real Alibaba data ────────────────────────────────────────
df      = pd.read_parquet(os.path.join("data", "processed",
                                        "alibaba_processed.parquet"))
adapter = CloudAdapter()
print(f"\n  Loaded {len(df):,} real Alibaba tasks")
print(f"  Columns: {list(df.columns)}")

# ── ILP Solver ────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  ILP SOLVER RESULTS")
print("=" * 55)
print(f"  {'Tasks':>6} {'Resources':>10} {'Status':>12} "
      f"{'Makespan':>10} {'Cost':>10} {'Energy':>10} {'Time':>8}")
print("  " + "-" * 70)

for n_tasks, n_res in [(20,5), (50,10), (100,10), (200,20)]:
    r = solve_from_df(df, adapter, n_tasks=n_tasks, n_resources=n_res)
    print(f"  {n_tasks:>6} {n_res:>10} {r['status']:>12} "
          f"{r['makespan']:>10.2f} {r['total_cost']:>10.4f} "
          f"{r['energy']:>10.4f} {r['solve_time']:>8.3f}s")

# ── GBR Predictive Model ──────────────────────────────────────────
print("\n" + "=" * 55)
print("  GBR PREDICTIVE MODEL RESULTS")
print("=" * 55)
print("  Training on real Alibaba data ...")
results = train(df)
print(f"  Windows used     : {results['n_windows']}")
print(f"  Arrival Rate MAE : {results['arrival_metrics']['MAE']}")
print(f"  Arrival Rate R²  : {results['arrival_metrics']['R2']}")
print(f"  Duration MAE     : {results['duration_metrics']['MAE']} seconds")
print(f"  Duration R²      : {results['duration_metrics']['R2']}")

print("\n  Forecasting next window ...")
forecast = predict(df.tail(3000))
print(f"  Expected arrivals : {forecast['arrival_rate']} tasks/window")
print(f"  Expected duration : {forecast['mean_duration']} seconds")

# ── PPO DRL Agent ─────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  PPO DRL AGENT RESULTS")
print("=" * 55)
print("  Loading trained PPO model ...")
model = load()
print("  Running episode on real Alibaba data ...")
r_ppo = run_episode(
    model, df, adapter,
    n_tasks=50, n_resources=10,
    forecast=forecast
)
print(f"  Tasks completed  : {r_ppo['tasks_done']}")
print(f"  SLA violations   : {r_ppo['sla_violations']}")
print(f"  Total energy     : {r_ppo['total_energy']} kWh")
print(f"  Total reward     : {r_ppo['total_reward']}")

# ── Knowledge Graph Validation ────────────────────────────────────
print("\n" + "=" * 55)
print("  KNOWLEDGE GRAPH VALIDATION")
print("=" * 55)
kg     = KnowledgeGraph()
result = solve_from_df(df, adapter, n_tasks=50, n_resources=5)
sample      = df.sample(n=50, random_state=42)
tasks       = [adapter.map_task(row) for row in sample.to_dict("records")]
machine_ids = sample["machine_id"].unique()[:5]
resources   = [adapter.map_resource({"machine_id": mid})
               for mid in machine_ids]
validation  = kg.validate_schedule(
    result["schedule"], tasks, resources
)
print(f"  Total checked : {validation['total']}")
print(f"  Valid         : {validation['valid']}")
print(f"  Invalid       : {validation['invalid']}")

print("\n" + "=" * 55)
print("  Real Alibaba evaluation complete.")
print("=" * 55)