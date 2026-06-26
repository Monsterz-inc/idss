"""
idss/adapters/cloud_adapter.py
-------------------------------
Maps Alibaba Cluster Trace v2018 records to AbstractTask / AbstractResource.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from schema import AbstractTask, AbstractResource
from adapters.base_adapter import BaseAdapter


class CloudAdapter(BaseAdapter):

    _CPU_CAP    = 1.0
    _MEM_CAP    = 1.0
    _COST_PER_S = 0.002
    _POWER_W    = 300.0

    def map_task(self, raw: dict) -> AbstractTask:
        start    = float(raw.get("start_time", 0))
        end      = float(raw.get("end_time", start + 1))
        dur      = max(end - start, 0.1)
        cpu      = min(float(raw.get("cpu_request", 0.1)), 1.0)
        mem      = min(float(raw.get("mem_request", 0.1)), 1.0)

        status_priority = {"Success": 2, "Failed": 4, "Killed": 3}
        priority = status_priority.get(str(raw.get("status", "Success")), 2)

        return AbstractTask(
            task_id       = str(raw.get("task_id", raw.get("job_id", "unknown"))),
            duration      = dur,
            cpu_demand    = cpu,
            memory_demand = mem,
            priority      = priority,
            deadline      = start + dur * 2.5,
            arrival_time  = start,
            domain        = "cloud",
            job_type      = str(raw.get("job_type", "batch")),
        )

    def map_resource(self, raw: dict) -> AbstractResource:
        return AbstractResource(
            resource_id     = str(raw.get("machine_id", "m0")),
            cpu_capacity    = self._CPU_CAP,
            memory_capacity = self._MEM_CAP,
            cost_per_second = self._COST_PER_S,
            power_watts     = self._POWER_W,
            available       = True,
        )

    def get_objective_weights(self):
        return {"makespan": 0.5, "cost": 0.25, "energy": 0.25}