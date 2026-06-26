"""
idss/models/drl/drl_env.py
---------------------------
Custom Gymnasium environment for the IDSS scheduling problem.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from schema import AbstractTask, AbstractResource, ScheduleEntry


class IDSSSchedulingEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(self, tasks, resources, forecast=None, max_steps=500):
        super().__init__()

        self.all_tasks   = list(tasks)
        self.resources   = list(resources)
        self.forecast    = forecast or {"arrival_rate": 10, "mean_duration": 45}
        self.max_steps   = max_steps
        self.n_resources = len(resources)

        n_task_features     = 5
        n_resource_features = 2
        n_forecast_features = 2

        obs_size = (n_task_features +
                    self.n_resources * n_resource_features +
                    n_forecast_features)

        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(obs_size,),
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(self.n_resources + 1)

        # Reward weights — stronger completion incentive
        self.w_delay      = 0.2
        self.w_energy     = 0.1
        self.w_sla        = 1.0
        self.w_balance    = 0.2
        self.w_completion = 2.0   # strong reward for completing a task
        self.w_defer      = 0.5   # strong penalty for deferring

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.task_queue     = list(self.all_tasks)
        self.current_time   = 0.0
        self.step_count     = 0
        self.schedule       = []
        self.resource_load  = {r.resource_id: 0.0 for r in self.resources}
        self.total_energy   = 0.0
        self.sla_violations = 0
        return self._get_obs(), {}

    def _get_obs(self):
        obs = []

        if self.task_queue:
            t = self.task_queue[0]
            deadline_slack = max(0, t.deadline - self.current_time)
            obs += [
                min(t.duration / 600.0,    1.0),
                min(t.cpu_demand,          1.0),
                min(t.memory_demand,       1.0),
                min(t.priority / 5.0,      1.0),
                min(deadline_slack / 3600, 1.0),
            ]
        else:
            obs += [0.0, 0.0, 0.0, 0.0, 0.0]

        for r in self.resources:
            load  = self.resource_load[r.resource_id]
            avail = 1.0 if r.available else 0.0
            obs  += [min(load, 1.0), avail]

        obs += [
            min(self.forecast["arrival_rate"]  / 500.0, 1.0),
            min(self.forecast["mean_duration"] / 600.0, 1.0),
        ]

        return np.array(obs, dtype=np.float32)

    def step(self, action):
        self.step_count += 1
        reward    = 0.0
        truncated = self.step_count >= self.max_steps

        if not self.task_queue:
            return self._get_obs(), 0.0, True, truncated, {}

        task = self.task_queue[0]

        # ── Defer action ──────────────────────────────────────────
        if action == self.n_resources:
            self.current_time += 1.0
            reward -= self.w_defer   # strong defer penalty
            return self._get_obs(), reward, False, truncated, {}

        # ── Assign action ─────────────────────────────────────────
        resource = self.resources[action]

        if not resource.available:
            reward -= 2.0
            return self._get_obs(), reward, False, truncated, {}

        # Schedule the task
        start = max(self.current_time, task.arrival_time)
        end   = start + task.duration

        self.schedule.append(ScheduleEntry(
            task_id     = task.task_id,
            resource_id = resource.resource_id,
            start_time  = round(start, 3),
            end_time    = round(end,   3),
            valid       = True,
        ))

        self.resource_load[resource.resource_id] += task.cpu_demand
        self.current_time = end

        # ── Reward components ─────────────────────────────────────
        # Big reward for completing a task
        reward += self.w_completion

        # Penalise delay
        delay   = max(0, end - task.arrival_time)
        reward -= self.w_delay * min(delay / 600.0, 1.0)

        # Penalise energy
        energy  = task.duration * resource.power_watts / 3600.0
        self.total_energy += energy
        reward -= self.w_energy * min(energy / 100.0, 1.0)

        # Penalise SLA violation
        if end > task.deadline:
            self.sla_violations += 1
            reward -= self.w_sla

        # Reward load balance
        loads    = list(self.resource_load.values())
        load_var = np.var(loads)
        reward  += self.w_balance * (1.0 - min(load_var, 1.0))

        self.task_queue.pop(0)

        terminated = len(self.task_queue) == 0
        return self._get_obs(), reward, terminated, truncated, {}

    def get_results(self):
        return {
            "schedule":       self.schedule,
            "total_energy":   round(self.total_energy, 4),
            "sla_violations": self.sla_violations,
            "tasks_done":     len(self.schedule),
            "current_time":   round(self.current_time, 3),
        }