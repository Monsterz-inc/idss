"""
idss/kg/kg_validator.py
------------------------
Knowledge Graph built with NetworkX.
Stores domain rules and validates scheduling actions
before they are committed.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import networkx as nx
from schema import AbstractTask, AbstractResource


class KnowledgeGraph:

    def __init__(self):
        self.G = nx.DiGraph()
        self._build_default_rules()

    def _build_default_rules(self):
        """
        Populate the graph with default cross-domain rules.
        Nodes: Task types, Resource types, Rule nodes
        Edges: CONFLICTS_WITH, REQUIRES, GOVERNED_BY
        """
        # ── Resource nodes ──────────────────────────────────────────
        self.G.add_node("cloud_resource",         type="resource")
        self.G.add_node("manufacturing_machine",  type="resource")
        self.G.add_node("logistics_vehicle",      type="resource")

        # ── Task type nodes ─────────────────────────────────────────
        self.G.add_node("batch_job",    type="task")
        self.G.add_node("online_job",   type="task")
        self.G.add_node("failed_task",  type="task")

        # ── Rule nodes ──────────────────────────────────────────────
        self.G.add_node("R1", type="rule",
            description="CPU demand must not exceed 0.95 per task")
        self.G.add_node("R2", type="rule",
            description="Failed tasks must be assigned higher priority")
        self.G.add_node("R3", type="rule",
            description="Online jobs cannot be assigned to unavailable resources")
        self.G.add_node("R4", type="rule",
            description="Task deadline must not already be exceeded at assignment")
        self.G.add_node("R5", type="rule",
            description="Memory demand must not exceed 0.95 per task")

        # ── Edges ───────────────────────────────────────────────────
        self.G.add_edge("batch_job",   "cloud_resource",        relation="REQUIRES")
        self.G.add_edge("online_job",  "cloud_resource",        relation="REQUIRES")
        self.G.add_edge("batch_job",   "R1",                    relation="GOVERNED_BY")
        self.G.add_edge("batch_job",   "R5",                    relation="GOVERNED_BY")
        self.G.add_edge("online_job",  "R3",                    relation="GOVERNED_BY")
        self.G.add_edge("failed_task", "R2",                    relation="GOVERNED_BY")
        self.G.add_edge("batch_job",   "R4",                    relation="GOVERNED_BY")
        self.G.add_edge("online_job",  "R4",                    relation="GOVERNED_BY")

    def validate(
        self,
        task: AbstractTask,
        resource: AbstractResource,
        current_time: float = 0.0,
    ) -> dict:
        """
        Validate a proposed (task, resource) assignment.
        Returns dict with keys:
            'valid'      : bool
            'violations' : list of rule IDs violated
            'messages'   : list of human-readable messages
        """
        violations = []
        messages   = []

        # R1 — CPU demand check
        if task.cpu_demand > 0.95:
            violations.append("R1")
            messages.append(
                f"R1: Task {task.task_id} CPU demand "
                f"({task.cpu_demand:.2f}) exceeds 0.95 limit."
            )

        # R5 — Memory demand check
        if task.memory_demand > 0.95:
            violations.append("R5")
            messages.append(
                f"R5: Task {task.task_id} memory demand "
                f"({task.memory_demand:.2f}) exceeds 0.95 limit."
            )

        # R3 — Online job on unavailable resource
        if task.job_type == "online" and not resource.available:
            violations.append("R3")
            messages.append(
                f"R3: Online task {task.task_id} cannot be assigned "
                f"to unavailable resource {resource.resource_id}."
            )

        # R4 — Deadline already exceeded
        if task.deadline < current_time:
            violations.append("R4")
            messages.append(
                f"R4: Task {task.task_id} deadline ({task.deadline:.1f}s) "
                f"already passed at current time ({current_time:.1f}s)."
            )

        # R2 — Failed task priority check
        if task.job_type == "batch" and task.priority < 3:
            # Failed tasks (mapped to priority 4) are fine
            # but if a failed task somehow got low priority, flag it
            pass

        return {
            "valid":      len(violations) == 0,
            "violations": violations,
            "messages":   messages,
        }

    def validate_schedule(self, schedule, tasks, resources, current_time=0.0):
        """
        Validate an entire schedule (list of ScheduleEntry).
        Returns summary dict.
        """
        task_map     = {t.task_id:     t for t in tasks}
        resource_map = {r.resource_id: r for r in resources}

        results   = []
        n_valid   = 0
        n_invalid = 0

        for entry in schedule:
            task     = task_map.get(entry.task_id)
            resource = resource_map.get(entry.resource_id)
            if task is None or resource is None:
                continue

            result = self.validate(task, resource, current_time)
            results.append({
                "task_id":    entry.task_id,
                "resource_id": entry.resource_id,
                **result,
            })
            if result["valid"]:
                n_valid += 1
            else:
                n_invalid += 1

        return {
            "total":    len(results),
            "valid":    n_valid,
            "invalid":  n_invalid,
            "details":  results,
        }

    def get_rules(self):
        """Return all rule nodes and their descriptions."""
        return {
            n: d["description"]
            for n, d in self.G.nodes(data=True)
            if d.get("type") == "rule"
        }

    def stats(self):
        return {
            "nodes": self.G.number_of_nodes(),
            "edges": self.G.number_of_edges(),
            "rules": len(self.get_rules()),
        }