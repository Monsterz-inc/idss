"""
idss/test_kg.py
----------------
Tests the Knowledge Graph validator.
Run from the idss folder:
    python test_kg.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from adapters.cloud_adapter import CloudAdapter
from models.optimisation.ilp_solver import solve_from_df
from kg.kg_validator import KnowledgeGraph

print("=" * 55)
print("  IDSS — Knowledge Graph Validator Test")
print("=" * 55)

# ── Build KG ─────────────────────────────────────────────────────
kg = KnowledgeGraph()
stats = kg.stats()
print(f"\n  KG built successfully")
print(f"  Nodes : {stats['nodes']}")
print(f"  Edges : {stats['edges']}")
print(f"  Rules : {stats['rules']}")

# ── Print all rules ───────────────────────────────────────────────
print("\n  Registered Rules:")
for rule_id, desc in kg.get_rules().items():
    print(f"    [{rule_id}] {desc}")

# ── Load data and generate a schedule ────────────────────────────
print("\n  Generating schedule to validate ...")
DATA_PATH = os.path.join("data", "processed", "synthetic_low_load.parquet")
df      = pd.read_parquet(DATA_PATH)
adapter = CloudAdapter()

result = solve_from_df(df, adapter, n_tasks=50, n_resources=5)
schedule  = result["schedule"]
print(f"  Schedule contains {len(schedule)} entries")

# Reconstruct tasks and resources for validation
sample      = df.sample(n=50, random_state=42)
tasks       = [adapter.map_task(row) for row in sample.to_dict("records")]
machine_ids = sample["machine_id"].unique()[:5]
resources   = [adapter.map_resource({"machine_id": mid}) for mid in machine_ids]

# ── Validate full schedule ────────────────────────────────────────
print("\n  Validating schedule against KG rules ...")
validation = kg.validate_schedule(schedule, tasks, resources, current_time=0.0)

print(f"\n  Results:")
print(f"    Total entries checked : {validation['total']}")
print(f"    Valid assignments     : {validation['valid']}")
print(f"    Invalid assignments   : {validation['invalid']}")

# ── Test individual rule violations ──────────────────────────────
print("\n  Testing individual rule violations:")

from schema import AbstractTask, AbstractResource

# R1 violation — CPU too high
bad_task = AbstractTask(
    task_id="bad_cpu", duration=10, cpu_demand=0.99,
    memory_demand=0.1, priority=2, deadline=9999,
    arrival_time=0, domain="cloud", job_type="batch"
)
good_resource = AbstractResource(
    resource_id="r0", cpu_capacity=1.0, memory_capacity=1.0,
    cost_per_second=0.002, power_watts=300, available=True
)
r1 = kg.validate(bad_task, good_resource)
print(f"\n  [R1 CPU test] Valid: {r1['valid']}")
print(f"    Message: {r1['messages'][0] if r1['messages'] else 'None'}")

# R3 violation — online job on unavailable resource
online_task = AbstractTask(
    task_id="online_001", duration=5, cpu_demand=0.2,
    memory_demand=0.1, priority=3, deadline=9999,
    arrival_time=0, domain="cloud", job_type="online"
)
unavail_resource = AbstractResource(
    resource_id="r1", cpu_capacity=1.0, memory_capacity=1.0,
    cost_per_second=0.002, power_watts=300, available=False
)
r3 = kg.validate(online_task, unavail_resource)
print(f"\n  [R3 Availability test] Valid: {r3['valid']}")
print(f"    Message: {r3['messages'][0] if r3['messages'] else 'None'}")

# R4 violation — deadline already passed
late_task = AbstractTask(
    task_id="late_001", duration=10, cpu_demand=0.2,
    memory_demand=0.1, priority=2, deadline=50.0,
    arrival_time=0, domain="cloud", job_type="batch"
)
r4 = kg.validate(late_task, good_resource, current_time=100.0)
print(f"\n  [R4 Deadline test] Valid: {r4['valid']}")
print(f"    Message: {r4['messages'][0] if r4['messages'] else 'None'}")

# Clean assignment — should pass all rules
clean_task = AbstractTask(
    task_id="clean_001", duration=10, cpu_demand=0.3,
    memory_demand=0.2, priority=2, deadline=9999,
    arrival_time=0, domain="cloud", job_type="batch"
)
clean = kg.validate(clean_task, good_resource)
print(f"\n  [Clean assignment test] Valid: {clean['valid']}")
print(f"    Violations: {clean['violations']}")

print("\n" + "=" * 55)
print("  Knowledge Graph tests complete.")
print("=" * 55)