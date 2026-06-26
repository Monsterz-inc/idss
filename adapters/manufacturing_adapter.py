"""
idss/adapters/manufacturing_adapter.py
---------------------------------------
Maps synthetic factory job records to AbstractTask / AbstractResource.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from schema import AbstractTask, AbstractResource
from adapters.base_adapter import BaseAdapter


class ManufacturingAdapter(BaseAdapter):

    _CPU_CAP    = 1.0
    _MEM_CAP    = 0.5
    _COST_PER_S = 0.005
    _POWER_W    = 50.0

    def map_task(self, raw: dict) -> AbstractTask:
        return AbstractTask(
            task_id       = str(raw.get("job_id", "j0")),
            duration      = float(raw.get("processing_time", 10.0)),
            cpu_demand    = 0.8,
            memory_demand = 0.2,
            priority      = int(raw.get("priority", 2)),
            deadline      = float(raw.get("due_date", 9999.0)),
            arrival_time  = float(raw.get("arrival_time", 0.0)),
            domain        = "manufacturing",
            job_type      = "batch",
        )

    def map_resource(self, raw: dict) -> AbstractResource:
        return AbstractResource(
            resource_id     = str(raw.get("machine_id", "mach0")),
            cpu_capacity    = self._CPU_CAP,
            memory_capacity = self._MEM_CAP,
            cost_per_second = self._COST_PER_S,
            power_watts     = self._POWER_W,
            available       = True,
        )

    def get_objective_weights(self):
        return {"makespan": 0.6, "cost": 0.2, "energy": 0.2}