"""
idss/schema.py
--------------
Abstract data classes that every Domain Adapter must produce.
These are the only objects that flow between layers internally.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AbstractTask:
    task_id: str
    duration: float          # expected execution time (seconds)
    cpu_demand: float        # fraction of one resource's CPU (0.0 - 1.0)
    memory_demand: float     # fraction of one resource's memory (0.0 - 1.0)
    priority: int            # 1 (low) to 5 (high)
    deadline: float          # absolute deadline (seconds from start)
    arrival_time: float      # when the task arrived (seconds from start)
    dependencies: List[str] = field(default_factory=list)
    domain: str = "cloud"
    job_type: str = "batch"


@dataclass
class AbstractResource:
    resource_id: str
    cpu_capacity: float
    memory_capacity: float
    cost_per_second: float
    power_watts: float
    available: bool = True


@dataclass
class ScheduleEntry:
    task_id: str
    resource_id: str
    start_time: float
    end_time: float
    valid: bool = True
    explanation: Optional[dict] = None