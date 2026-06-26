"""
idss/adapters/logistics_adapter.py
------------------------------------
Maps delivery/vehicle routing records to AbstractTask / AbstractResource.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from schema import AbstractTask, AbstractResource
from adapters.base_adapter import BaseAdapter

AVG_SPEED_KMH = 60.0


class LogisticsAdapter(BaseAdapter):

    _CPU_CAP    = 1.0
    _MEM_CAP    = 0.3
    _COST_PER_S = 0.008
    _POWER_W    = 80.0

    def map_task(self, raw: dict) -> AbstractTask:
        dist    = float(raw.get("distance_km", 10.0))
        dur     = (dist / AVG_SPEED_KMH) * 3600
        load    = float(raw.get("load_kg", 100.0))
        cpu_dem = min(load / 1000.0, 1.0)

        return AbstractTask(
            task_id       = str(raw.get("delivery_id", "d0")),
            duration      = dur,
            cpu_demand    = cpu_dem,
            memory_demand = 0.1,
            priority      = 3,
            deadline      = float(raw.get("time_window_end", 9999.0)),
            arrival_time  = float(raw.get("arrival_time", 0.0)),
            domain        = "logistics",
            job_type      = "online",
        )

    def map_resource(self, raw: dict) -> AbstractResource:
        return AbstractResource(
            resource_id     = str(raw.get("vehicle_id", "v0")),
            cpu_capacity    = self._CPU_CAP,
            memory_capacity = self._MEM_CAP,
            cost_per_second = self._COST_PER_S,
            power_watts     = self._POWER_W,
            available       = True,
        )

    def get_objective_weights(self):
        return {"makespan": 0.4, "cost": 0.4, "energy": 0.2}