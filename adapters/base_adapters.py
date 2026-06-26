"""
idss/adapters/base_adapter.py
------------------------------
Abstract base class every Domain Adapter must implement.
"""
from abc import ABC, abstractmethod
from typing import List, Dict
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from schema import AbstractTask, AbstractResource


class BaseAdapter(ABC):

    @abstractmethod
    def map_task(self, raw: dict) -> AbstractTask:
        """Convert a raw domain record dict to AbstractTask."""
        pass

    @abstractmethod
    def map_resource(self, raw: dict) -> AbstractResource:
        """Convert a raw resource record dict to AbstractResource."""
        pass

    @abstractmethod
    def get_objective_weights(self) -> Dict[str, float]:
        """
        Return weights for the multi-objective function.
        Keys: 'makespan', 'cost', 'energy'
        Values: floats that sum to 1.0
        """
        pass