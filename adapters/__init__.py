from adapters.cloud_adapter import CloudAdapter
from adapters.manufacturing_adapter import ManufacturingAdapter
from adapters.logistics_adapter import LogisticsAdapter

ADAPTERS = {
    "cloud": CloudAdapter,
    "manufacturing": ManufacturingAdapter,
    "logistics": LogisticsAdapter,
}

def get_adapter(domain: str):
    cls = ADAPTERS.get(domain.lower())
    if cls is None:
        raise ValueError(f"Unknown domain '{domain}'. Choose from: {list(ADAPTERS.keys())}")
    return cls()