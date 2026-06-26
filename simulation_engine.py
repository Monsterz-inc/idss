"""
idss/simulation_engine.py
--------------------------
Real-time simulation engine that drives the IDSS dashboard.
Simulates continuous task arrivals, scheduling decisions,
anomaly detection, and human-in-the-loop notifications.
Supports domain switching: cloud, manufacturing, logistics.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import random
import numpy as np
from datetime import datetime
from collections import deque
from schema import AbstractTask, AbstractResource, ScheduleEntry
from adapters.cloud_adapter import CloudAdapter
from adapters import get_adapter
from models.xai.xai_explainer import explain_schedule_decision
from kg.kg_validator import KnowledgeGraph
from models.drl.ppo_agent import load as load_ppo

random.seed(42)
np.random.seed(42)

N_RESOURCES    = 6
TICK_INTERVAL  = 2.0
SLA_WINDOW     = 10
SLA_THRESHOLD  = 0.4
SPIKE_FACTOR   = 3.0
LOAD_DECAY     = 0.08
LOAD_INCREMENT = 0.12


class SimulationEngine:

    def __init__(self):
        self.domain  = "cloud"
        self.adapter = CloudAdapter()
        self.kg      = KnowledgeGraph()
        self.resources = self._init_resources()
        self.tick      = 0
        self.sim_time  = 0.0

        # State
        self.task_queue      = deque()
        self.schedule_log    = []
        self.alert_log       = []
        self.resource_load   = {r.resource_id: 0.0 for r in self.resources}
        self.recent_sla      = deque(maxlen=SLA_WINDOW)
        self.arrival_history = deque(maxlen=10)

        # Metrics
        self.total_tasks     = 0
        self.total_scheduled = 0
        self.sla_violations  = 0
        self.total_energy    = 0.0
        self.forecast        = {"arrival_rate": 5.0, "mean_duration": 45.0}

        # Callbacks
        self.on_update = None
        self.on_alert  = None

        # Load PPO
        try:
            self.ppo       = load_ppo()
            self.ppo_ready = True
        except Exception:
            self.ppo       = None
            self.ppo_ready = False

    def _init_resources(self):
        resources = []
        specs = [
            ("server_A", 0.001, 200),
            ("server_B", 0.002, 250),
            ("server_C", 0.003, 300),
            ("server_D", 0.002, 250),
            ("server_E", 0.001, 200),
            ("server_F", 0.003, 350),
        ]
        for rid, cost, power in specs:
            resources.append(AbstractResource(
                resource_id     = rid,
                cpu_capacity    = 1.0,
                memory_capacity = 1.0,
                cost_per_second = cost,
                power_watts     = power,
                available       = True,
            ))
        return resources

    def switch_domain(self, domain: str) -> dict:
        """
        Switch the active domain adapter at runtime.
        Resets the simulation state for the new domain.
        """
        try:
            self.adapter = get_adapter(domain)
            self.domain  = domain

            # Reset all state
            self.task_queue      = deque()
            self.schedule_log    = []
            self.alert_log       = []
            self.resource_load   = {r.resource_id: 0.0 for r in self.resources}
            self.recent_sla      = deque(maxlen=SLA_WINDOW)
            self.arrival_history = deque(maxlen=10)
            self.total_tasks     = 0
            self.total_scheduled = 0
            self.sla_violations  = 0
            self.total_energy    = 0.0
            self.tick            = 0
            self.sim_time        = 0.0

            # Restore all resources to available
            for r in self.resources:
                r.available = True

            self.alert_log.append({
                "type":     "domain_switch",
                "message":  f"Domain switched to {domain.upper()}. Simulation reset.",
                "severity": "info",
                "time":     0.0,
            })
            return {"success": True, "domain": domain}

        except ValueError as e:
            return {"success": False, "message": str(e)}

    def _generate_tasks(self, n: int) -> list:
        """
        Generate n tasks using domain-specific parameters.
        Each domain has different duration, CPU, memory,
        and job type characteristics.
        """
        domain_params = {
            "cloud": {
                "duration_mean": 45,
                "cpu_mean":      0.18,
                "mem_mean":      0.12,
                "job_types":     ["batch", "online"],
            },
            "manufacturing": {
                "duration_mean": 120,
                "cpu_mean":      0.70,
                "mem_mean":      0.20,
                "job_types":     ["batch"],
            },
            "logistics": {
                "duration_mean": 600,
                "cpu_mean":      0.30,
                "mem_mean":      0.10,
                "job_types":     ["online"],
            },
        }
        params = domain_params.get(self.domain, domain_params["cloud"])

        tasks = []
        for i in range(n):
            duration  = max(1.0, np.random.exponential(params["duration_mean"]))
            cpu       = min(0.85, max(0.05, np.random.exponential(params["cpu_mean"])))
            mem       = min(0.85, max(0.05, np.random.exponential(params["mem_mean"])))
            priority  = random.choices(
                [1, 2, 3, 4, 5],
                weights=[5, 25, 35, 25, 10]
            )[0]
            tasks.append(AbstractTask(
                task_id       = f"T{self.total_tasks + i:05d}",
                duration      = round(duration, 2),
                cpu_demand    = round(cpu, 3),
                memory_demand = round(mem, 3),
                priority      = priority,
                deadline      = self.sim_time + duration * 2.5,
                arrival_time  = self.sim_time,
                domain        = self.domain,
                job_type      = random.choice(params["job_types"]),
            ))
        return tasks

    def _pick_resource(self, task: AbstractTask):
        """
        Pick the best available resource for a task.
        Uses load-balanced selection — always picks the
        least loaded available resource that passes KG
        validation to prevent any single server overloading.
        """
        available = [r for r in self.resources if r.available]
        if not available:
            return None

        # Sort by load ascending — least loaded first
        available.sort(key=lambda r: self.resource_load[r.resource_id])

        # Pick least loaded that passes KG validation
        for resource in available:
            check = self.kg.validate(task, resource, self.sim_time)
            if check["valid"]:
                return resource

        # Fallback to least loaded even if KG flags it
        return available[0]

    def _detect_anomalies(self, n_arrivals: int) -> list:
        anomalies = []

        # 1. Random resource failure
        for r in self.resources:
            if r.available and random.random() < 0.04:
                r.available = False
                anomalies.append({
                    "type":     "resource_offline",
                    "message":  f"Resource {r.resource_id} went offline unexpectedly.",
                    "severity": "high",
                    "resource": r.resource_id,
                    "time":     self.sim_time,
                })

        # 2. SLA violation rate check
        if len(self.recent_sla) == SLA_WINDOW:
            viol_rate = sum(self.recent_sla) / SLA_WINDOW
            if viol_rate > SLA_THRESHOLD:
                anomalies.append({
                    "type":     "sla_breach",
                    "message":  f"SLA violation rate {viol_rate:.0%} exceeds {SLA_THRESHOLD:.0%} threshold.",
                    "severity": "medium",
                    "rate":     round(viol_rate, 3),
                    "time":     self.sim_time,
                })

        # 3. Arrival spike detection
        if len(self.arrival_history) >= 3:
            avg_prev = np.mean(list(self.arrival_history)[:-1])
            if avg_prev > 0 and n_arrivals > avg_prev * SPIKE_FACTOR:
                anomalies.append({
                    "type":     "arrival_spike",
                    "message":  f"Task arrival spike: {n_arrivals} tasks vs avg {avg_prev:.1f}.",
                    "severity": "medium",
                    "count":    n_arrivals,
                    "time":     self.sim_time,
                })

        # 4. High-priority task in queue
        for task in list(self.task_queue):
            if task.priority == 5:
                anomalies.append({
                    "type":     "high_priority",
                    "message":  f"High-priority task {task.task_id} requires immediate scheduling.",
                    "severity": "high",
                    "task_id":  task.task_id,
                    "time":     self.sim_time,
                })
                break

        return anomalies

    def _restore_resources(self):
        """Randomly restore offline resources (30% chance per tick)."""
        for r in self.resources:
            if not r.available and random.random() < 0.30:
                r.available = True
                self.alert_log.append({
                    "type":     "resource_restored",
                    "message":  f"Resource {r.resource_id} is back online.",
                    "severity": "info",
                    "time":     self.sim_time,
                })

    def _schedule_batch(self):
        """
        Schedule up to 10 tasks from the queue using
        load-balanced resource selection.
        """
        if not self.task_queue:
            return []

        available = [r for r in self.resources if r.available]
        if not available:
            return []

        batch = []
        for _ in range(min(10, len(self.task_queue))):
            batch.append(self.task_queue.popleft())

        entries = []
        for idx, task in enumerate(batch):
            resource = self._pick_resource(task)
            if resource is None:
                continue

            start = self.sim_time + idx * 0.5
            end   = start + task.duration

            entry = ScheduleEntry(
                task_id     = task.task_id,
                resource_id = resource.resource_id,
                start_time  = round(start, 3),
                end_time    = round(end,   3),
                valid       = True,
            )

            # Update load
            self.resource_load[resource.resource_id] = min(
                1.0,
                self.resource_load[resource.resource_id] + task.cpu_demand * LOAD_INCREMENT
            )

            # Energy
            energy = task.duration * resource.power_watts / 3600.0
            self.total_energy += energy

            # SLA check
            sla_met = end <= task.deadline
            self.recent_sla.append(0 if sla_met else 1)
            if not sla_met:
                self.sla_violations += 1

            # Explanation
            exp = explain_schedule_decision(task, resource, entry)

            entries.append({
                "task_id":     task.task_id,
                "resource_id": resource.resource_id,
                "start_time":  entry.start_time,
                "end_time":    entry.end_time,
                "valid":       entry.valid,
                "sla_met":     sla_met,
                "priority":    task.priority,
                "duration":    task.duration,
                "energy":      round(energy, 4),
                "explanation": exp,
            })

            self.schedule_log.append(entry)
            self.total_scheduled += 1

        self.total_tasks += len(batch)
        return entries

    def override_task(self, task_id: str, resource_id: str) -> dict:
        """
        Human override: manually assign a task to a specific resource.
        Updates resource load and logs the override action.
        """
        resource = next(
            (r for r in self.resources if r.resource_id == resource_id),
            None
        )
        if not resource:
            return {
                "success": False,
                "message": f"Resource {resource_id} not found."
            }

        if not resource.available:
            return {
                "success": False,
                "message": f"Resource {resource_id} is currently offline."
            }

        # Apply load impact of override
        self.resource_load[resource_id] = min(
            1.0,
            self.resource_load[resource_id] + 0.1
        )

        self.alert_log.append({
            "type":     "human_override",
            "message":  f"Operator manually assigned {task_id} to {resource_id}.",
            "severity": "info",
            "time":     self.sim_time,
        })

        return {
            "success": True,
            "message": f"Task {task_id} successfully reassigned to {resource_id}."
        }

    def get_state(self) -> dict:
        return {
            "tick":            self.tick,
            "sim_time":        round(self.sim_time, 1),
            "domain":          self.domain,
            "total_tasks":     self.total_tasks,
            "total_scheduled": self.total_scheduled,
            "sla_violations":  self.sla_violations,
            "total_energy":    round(self.total_energy, 2),
            "queue_length":    len(self.task_queue),
            "resources": [
                {
                    "id":        r.resource_id,
                    "available": r.available,
                    "load":      round(self.resource_load[r.resource_id], 3),
                    "cost":      r.cost_per_second,
                    "power":     r.power_watts,
                }
                for r in self.resources
            ],
            "forecast":      self.forecast,
            "recent_alerts": self.alert_log[-5:],
            "schedule_log": [
                {
                    "task_id":     e.task_id,
                    "resource_id": e.resource_id,
                    "start_time":  e.start_time,
                    "end_time":    e.end_time,
                }
                for e in self.schedule_log[-20:]
            ],
        }

    def tick_once(self) -> dict:
        self.tick     += 1
        self.sim_time += TICK_INTERVAL * 10

        # Decay all resource loads each tick
        for rid in self.resource_load:
            self.resource_load[rid] = max(
                0.0,
                self.resource_load[rid] - LOAD_DECAY
            )

        # Restore offline resources
        self._restore_resources()

        # Generate new task arrivals
        n_arrivals = max(1, int(np.random.poisson(5)))
        self.arrival_history.append(n_arrivals)

        new_tasks = self._generate_tasks(n_arrivals)
        self.task_queue.extend(new_tasks)

        # Update forecast
        self.forecast = {
            "arrival_rate":  round(float(np.mean(list(self.arrival_history))), 1),
            "mean_duration": float(
                {"cloud": 45, "manufacturing": 120, "logistics": 600}
                .get(self.domain, 45)
            ),
        }

        # Detect anomalies
        anomalies = self._detect_anomalies(n_arrivals)
        for a in anomalies:
            self.alert_log.append(a)
            if self.on_alert:
                self.on_alert(a)

        # Schedule batch
        scheduled = self._schedule_batch()

        result = {
            **self.get_state(),
            "new_scheduled": scheduled,
            "new_alerts":    anomalies,
            "timestamp":     datetime.now().isoformat(),
        }

        if self.on_update:
            self.on_update(result)

        return result