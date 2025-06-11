from typing import Self
import numpy as np
import xarray as xr
from ..unit_handling import Quantity

class PointsConfig:

    def __init__(self, spec: dict[str, dict]) -> None:

        self.spec: dict[str, PointConfig] = dict()
        for key, point_spec in spec.items():
            self.spec[key] = PointConfig(point_spec)

class PointConfig:

    allowed_methods = ["minimize", "maximize", "nearest_to", "interp_to"]

    def __init__(self, spec: dict) -> None:
        
        method = [method for method in self.allowed_methods if method in spec.keys()]
        assert len(method) == 1, f"Must provide one of [{', '.join(self.allowed_methods)}] for a point. Keys were {list(spec.keys())}"

        self.method = method[0]

        self.mask = MaskConfig(spec.get("where", dict()))

class MaskConfig:

    def __init__(self, spec: dict) -> None:
        
        self.spec: dict[str, ConditionConfig] = dict()
        for key, mask_spec in spec.items():
            self.spec[key] = ConditionConfig.from_spec(key, mask_spec)
    
    def __call__(self, dataset: xr.Dataset) -> xr.DataArray:
        mask = xr.DataArray(True)
        
        for condition in self.spec.values():
            field = dataset[condition.key]
            mask = mask & ((field > condition.min) & (field < condition.max))
        
        return mask

class ConditionConfig:

    @classmethod
    def from_spec(cls, key: str, spec: dict) -> Self:
        min_val = Quantity(spec.get("min", -np.inf), spec.get("units", ""))
        max_val = Quantity(spec.get("max", +np.inf), spec.get("units", ""))

        return cls(key, min_val, max_val)
    
    def __init__(self, key: str, min_val: Quantity, max_val: Quantity) -> None:
        self.key = key
        self.min = min_val
        self.max = max_val
    