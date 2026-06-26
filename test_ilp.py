"""
idss/test_ilp.py
-----------------
Quick test of the ILP solver against synthetic data.
Run from the idss folder:
    python test_ilp.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from adapters.cloud_adapter import CloudAdapter
from models.optimisation.ilp_solver import solve_from_df

# ── Load synthetic data ───────────────────────────────────────────
DATA_PATH = os.path.join("data", "processed", "synthetic_low_load.parquet")

print("=" * 55)
print("  IDSS — ILP Solver Test")
print("=" * 55)

df = pd.read_parquet(DATA_PATH)
print(f"\n  Loaded {len(df):,} tasks from synthetic_low_load")

adapter = CloudAdapter()

# ── Test 1: Small instance (20 tasks, 5 resources) ───────────────
print("\n  [Test 1] 20 tasks | 5 resources")
print("  Solving ...", end=" ")
result = solve_from_df(df, adapter, n_tasks=20, n_resources=5)
print("Done.")
print(f"  Status     : {result['status']}")
print(f"  Makespan   : {result['makespan']:.2f} seconds")
print(f"  Total cost : {result['total_cost']:.4f} units")
print(f"  Energy     : {result['energy']:.4f} kWh")
print(f"  Solve time : {result['solve_time']:.3f} seconds")
print(f"  Tasks scheduled: {len(result['schedule'])}")

# ── Test 2: Medium instance (100 tasks, 10 resources) ────────────
print("\n  [Test 2] 100 tasks | 10 resources")
print("  Solving ...", end=" ")
result2 = solve_from_df(df, adapter, n_tasks=100, n_resources=10)
print("Done.")
print(f"  Status     : {result2['status']}")
print(f"  Makespan   : {result2['makespan']:.2f} seconds")
print(f"  Total cost : {result2['total_cost']:.4f} units")
print(f"  Energy     : {result2['energy']:.4f} kWh")
print(f"  Solve time : {result2['solve_time']:.3f} seconds")
print(f"  Tasks scheduled: {len(result2['schedule'])}")

# ── Test 3: Show first 5 schedule entries ─────────────────────────
print("\n  [Test 3] First 5 schedule entries from Test 2:")
print(f"  {'Task ID':<25} {'Resource':<12} {'Start':>8} {'End':>8}")
print("  " + "-" * 55)
for entry in result2['schedule'][:5]:
    print(f"  {entry.task_id:<25} {entry.resource_id:<12} "
          f"{entry.start_time:>8.2f} {entry.end_time:>8.2f}")

# ── Test 4: Larger instance (200 tasks, 20 resources) ────────────
print("\n  [Test 4] 200 tasks | 20 resources (60s time limit)")
print("  Solving ...", end=" ")
result3 = solve_from_df(df, adapter, n_tasks=200, n_resources=20)
print("Done.")
print(f"  Status     : {result3['status']}")
print(f"  Makespan   : {result3['makespan']:.2f} seconds")
print(f"  Solve time : {result3['solve_time']:.3f} seconds")
print(f"  Tasks scheduled: {len(result3['schedule'])}")

print("\n" + "=" * 55)
print("  ILP Solver tests complete.")
print("=" * 55)