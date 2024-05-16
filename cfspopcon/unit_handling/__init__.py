"""Uses pint and xarray to enable unit-handling over multi-dimensional arrays."""
from typing import Union

import xarray as xr
from pint import DimensionalityError, UndefinedUnitError, UnitStrippedWarning

from .decorator import wraps_ufunc
from .default_units import convert_to_default_units, default_unit, magnitude_in_default_units, set_default_units
from .setup_unit_handling import (
    Quantity,
    Unit,
    convert_units,
    dimensionless_magnitude,
    get_units,
    magnitude,
    magnitude_in_units,
    ureg,
)

Unitfull = Union[Quantity, xr.DataArray]

__all__ = [
    "ureg",
    "Quantity",
    "Unit",
    "Unitfull",
    "wraps_ufunc",
    "magnitude_in_default_units",
    "magnitude_in_units",
    "set_default_units",
    "default_unit",
    "convert_to_default_units",
    "convert_units",
    "magnitude",
    "dimensionless_magnitude",
    "UnitStrippedWarning",
    "UndefinedUnitError",
    "DimensionalityError",
    "get_units",
]
