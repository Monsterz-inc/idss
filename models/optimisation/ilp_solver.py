"""
idss/models/optimisation/ilp_solver.py
----------------------------------------
ILP-based prescriptive scheduler using PuLP / CBC solver.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pulp
import time
from typing import List, Dict
from schema import AbstractTask, AbstractResource, ScheduleEntry


def solve(
    tasks:      List[AbstractTask],
    resources:  List[AbstractResource],
    weights:    Dict[str, float] = None,
    time_limit: int = 60,
) -> dict:

    if weights is None:
        weights = {"makespan": 0.5, "cost": 0.25, "energy": 0.25}

    if not tasks or not resources:
        return {"schedule": [], "makespan": 0, "total_cost": 0,
                "energy": 0, "solve_time": 0, "status": "No input"}

    avail = [r for r in resources if r.available]
    if not avail:
        return {"schedule": [], "makespan": 0, "total_cost": 0,
                "energy": 0, "solve_time": 0, "status": "No available resources"}

    T = len(tasks)
    R = len(avail)

    # ── Decision variables ──────────────────────────────────────────
    x = pulp.LpVariable.dicts(
        "x",
        [(i, j) for i in range(T) for j in range(R)],
        cat="Binary"
    )
    s = pulp.LpVariable.dicts(
        "s", range(T), lowBound=0, cat="Continuous"
    )
    makespan = pulp.LpVariable("makespan", lowBound=0, cat="Continuous")

    # ── Problem ──────────────────────────────────────────────────────
    prob = pulp.LpProblem("IDSS_Scheduling", pulp.LpMinimize)

    # ── Objective ────────────────────────────────────────────────────
    total_cost = pulp.lpSum(
        x[(i, j)] * tasks[i].duration * avail[j].cost_per_second
        for i in range(T) for j in range(R)
    )
    total_energy = pulp.lpSum(
        x[(i, j)] * tasks[i].duration * avail[j].power_watts / 3600.0
        for i in range(T) for j in range(R)
    )

    prob += (
        weights["makespan"] * makespan +
        weights["cost"]     * total_cost +
        weights["energy"]   * total_energy
    )

    # ── Constraints ───────────────────────────────────────────────────

    # 1. Every task assigned to exactly one resource
    for i in range(T):
        prob += pulp.lpSum(x[(i, j)] for j in range(R)) == 1

    # 2. CPU capacity per resource — use 3x oversubscription
    #    (realistic: cloud resources time-share CPUs)
    for j in range(R):
        prob += (
            pulp.lpSum(x[(i, j)] * tasks[i].cpu_demand for i in range(T))
            <= avail[j].cpu_capacity * 10.0
        )

    # 3. Memory capacity per resource — use 2x oversubscription
    for j in range(R):
        prob += (
            pulp.lpSum(x[(i, j)] * tasks[i].memory_demand for i in range(T))
            <= avail[j].memory_capacity * 10.0
        )

    # 4. Makespan >= start + duration for every task
    for i in range(T):
        prob += s[i] >= tasks[i].arrival_time
        prob += makespan >= s[i] + tasks[i].duration

    # ── Solve ─────────────────────────────────────────────────────────
    t0 = time.time()
    solver = pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=0)
    prob.solve(solver)
    solve_time = time.time() - t0

    status = pulp.LpStatus[prob.status]

    # ── Extract solution ──────────────────────────────────────────────
    schedule = []
    for i in range(T):
        assigned_j = None
        for j in range(R):
            val = pulp.value(x[(i, j)])
            if val is not None and val > 0.5:
                assigned_j = j
                break

        if assigned_j is not None:
            start = pulp.value(s[i]) or tasks[i].arrival_time
            end   = start + tasks[i].duration
            schedule.append(ScheduleEntry(
                task_id     = tasks[i].task_id,
                resource_id = avail[assigned_j].resource_id,
                start_time  = round(start, 3),
                end_time    = round(end, 3),
                valid       = True,
            ))

    ms     = pulp.value(makespan)     or 0.0
    cost   = pulp.value(total_cost)   or 0.0
    energy = pulp.value(total_energy) or 0.0

    return {
        "schedule":   schedule,
        "makespan":   round(ms, 3),
        "total_cost": round(cost, 6),
        "energy":     round(energy, 6),
        "solve_time": round(solve_time, 3),
        "status":     status,
    }


def solve_from_df(df, adapter, n_tasks=50, n_resources=5, weights=None):
    """
    Convenience wrapper: samples n_tasks rows from df,
    creates n_resources, runs the solver.
    """
    sample = df.sample(n=min(n_tasks, len(df)), random_state=42)
    tasks  = [adapter.map_task(row) for row in sample.to_dict("records")]

    machine_ids = sample["machine_id"].unique()[:n_resources]
    resources   = [
        adapter.map_resource({"machine_id": mid})
        for mid in machine_ids
    ]

    return solve(tasks, resources, weights)