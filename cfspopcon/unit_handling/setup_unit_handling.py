"""Set up the pint library for unit handling."""
import warnings
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar, Union, overload

import numpy as np
import numpy.typing as npt
import pint
import pint_xarray  # type:ignore[import-untyped]
import xarray as xr
from pint.errors import DimensionalityError, UndefinedUnitError
from typing_extensions import ParamSpec

ureg = pint_xarray.setup_registry(
    pint.UnitRegistry(
        force_ndarray_like=True,
    )
)

Quantity = ureg.Quantity
Unit = ureg.Unit

Params = ParamSpec("Params")
Ret = TypeVar("Ret")

# Define custom units for density as n_19 or n_20 (used in several formulas)
ureg.define("_1e19_per_cubic_metre = 1e19 m^-3 = 1e19 m^-3 = n19")
ureg.define("_1e20_per_cubic_metre = 1e20 m^-3 = 1e10 m^-3 = n20")
ureg.define("percent = 0.01")

# Needed for serialization/deserialization
pint.set_application_registry(ureg)  # type:ignore[no-untyped-call]


def suppress_downcast_warning(func: Callable[Params, Ret]) -> Callable[Params, Ret]:
    """Suppresses a common warning about downcasting quantities to arrays."""

    @wraps(func)
    def wrapper(*args: Params.args, **kwargs: Params.kwargs) -> Ret:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The unit of the quantity is stripped when downcasting to ndarray.")
            return func(*args, **kwargs)

    return wrapper


@overload
def convert_units(array: xr.DataArray, units: Union[str, pint.Unit]) -> xr.DataArray:
    ...


@overload
def convert_units(array: pint.Quantity, units: Union[str, pint.Unit]) -> pint.Quantity:
    ...


def convert_units(array: Union[xr.DataArray, pint.Quantity], units: Any) -> Union[xr.DataArray, pint.Quantity]:
    """Convert an array to specified units, handling both Quantities and xr.DataArrays."""
    if isinstance(array, xr.DataArray):
        if not hasattr(array.pint, "units") or array.pint.units is None:
            array = array.pint.quantify(ureg.dimensionless)

        return array.pint.to(units)  # type: ignore[no-any-return]
    elif isinstance(array, Quantity):
        return array.to(units)  # type:ignore[no-any-return]
    else:
        raise NotImplementedError(f"No implementation for 'convert_units' with an array of type {type(array)} ({array})")


@suppress_downcast_warning
def magnitude(array: Union[xr.DataArray, pint.Quantity]) -> Union[npt.NDArray[np.float32], float]:
    """Return the magnitude of an array, handling both Quantities and xr.DataArrays."""
    if isinstance(array, xr.DataArray):
        return array.values
    elif isinstance(array, Quantity):
        return array.magnitude  # type: ignore[no-any-return]
    else:
        raise NotImplementedError(f"No implementation for 'magnitude' with an array of type {type(array)} ({array})")


def dimensionless_magnitude(array: Union[xr.DataArray, pint.Quantity]) -> Union[npt.NDArray[np.float32], float]:
    """Converts the array to dimensionless and returns the magnitude."""
    return magnitude(convert_units(array, ureg.dimensionless))


__all__ = [
    "DimensionalityError",
    "UndefinedUnitError",
    "ureg",
    "Quantity",
    "Unit",
    "suppress_downcast_warning",
    "convert_units",
    "magnitude",
    "suppress_downcast_warning",
    "dimensionless_magnitude",
]
