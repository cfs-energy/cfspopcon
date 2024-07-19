"""Define default units for writing to/from disk."""

from collections.abc import Iterable
from importlib.resources import as_file, files
from numbers import Number
from typing import Any, Union, overload
from warnings import warn

import numpy as np
import xarray as xr
import yaml
from pint import DimensionalityError, UndefinedUnitError

from .setup_unit_handling import Quantity, convert_units, magnitude_in_units


def check_units_are_valid(units_dictionary: dict[str, str]) -> None:
    """Ensure that all units in units_dictionary are valid."""
    invalid_units = []
    for key, units in units_dictionary.items():
        try:
            Quantity(1.0, units)
        except UndefinedUnitError:  # noqa: PERF203
            warn(f"Undefined units '{units}' for '{key}", stacklevel=3)
            invalid_units.append(units)

    if invalid_units:
        raise UndefinedUnitError(invalid_units)  # type:ignore[arg-type]


def read_default_units() -> dict[str, str]:
    """Read in the default_units.yaml file."""
    with as_file(files("cfspopcon").joinpath("default_units.yaml")) as filepath:
        with open(filepath) as f:
            units_dictionary: dict[str, str] = yaml.safe_load(f)

    check_units_are_valid(units_dictionary)
    return units_dictionary


DEFAULT_UNITS = read_default_units()


def default_unit(var: str) -> Union[str, None]:
    """Return cfspopcon's default unit for a given quantity.

    Args:
        var: Quantity name

    Returns: Unit
    """
    try:
        return DEFAULT_UNITS[var]
    except KeyError:
        raise KeyError(
            f"No default unit defined for {var}. Please check configured default units in the unit_handling submodule."
        ) from None


def magnitude_in_default_units(value: Union[Quantity, xr.DataArray], key: str) -> Union[float, list[float], Any]:
    """Convert values to default units and then return the magnitude.

    Args:
        value: input value to convert to a float
        key: name of field for looking up in DEFAULT_UNITS dictionary

    Returns:
        magnitude of value in default units and as basic type
    """
    try:
        # unit conversion step
        unit = default_unit(key)
        if unit is None:
            return value

        mag = magnitude_in_units(value, unit)

    except DimensionalityError as e:
        print(f"Unit conversion failed for {key}. Could not convert '{value}' to '{DEFAULT_UNITS[key]}'")
        raise e

    # single value arrays -> float
    # np,xr array -> list
    if isinstance(mag, (np.ndarray, xr.DataArray)):
        if mag.size == 1:
            return float(mag)
        else:
            return [float(v) for v in mag]
    else:
        return float(mag)


@overload
def set_default_units(value: Number, key: str) -> Quantity: ...


@overload
def set_default_units(value: xr.DataArray, key: str) -> xr.DataArray: ...


@overload
def set_default_units(value: Any, key: str) -> Any: ...


def set_default_units(value: Any, key: str) -> Any:
    """Return value as a quantity with default units.

    Args:
        value: magnitude of input value to convert to a Quantity
        key: name of field for looking up in DEFAULT_UNITS dictionary

    Returns:
        magnitude of value in default units
    """

    def _is_number_not_bool(val: Any) -> bool:
        return isinstance(val, Number) and not isinstance(val, bool)

    def _is_iterable_of_number_not_bool(val: Any) -> bool:
        if not isinstance(val, Iterable):
            return False

        if isinstance(val, (np.ndarray, xr.DataArray)) and val.ndim == 0:
            return _is_number_not_bool(val.item())

        return all(_is_number_not_bool(v) for v in value)

    # None is used to ignore class types
    if DEFAULT_UNITS[key] is None:
        if _is_number_not_bool(value) or _is_iterable_of_number_not_bool(value):
            raise RuntimeError(
                f"set_default_units for key {key} and value {value} of type {type(value)}: numeric types should carry units!"
            )
        return value
    elif isinstance(value, xr.DataArray):
        return value.pint.quantify(DEFAULT_UNITS[key])
    else:
        return Quantity(value, DEFAULT_UNITS[key])


@overload
def convert_to_default_units(value: float, key: str) -> float: ...


@overload
def convert_to_default_units(value: xr.DataArray, key: str) -> xr.DataArray: ...


@overload
def convert_to_default_units(value: Quantity, key: str) -> Quantity: ...


def convert_to_default_units(value: Union[float, Quantity, xr.DataArray], key: str) -> Union[float, Quantity, xr.DataArray]:
    """Convert an array or scalar to default units."""
    unit = DEFAULT_UNITS[key]
    if unit is None:
        return value
    elif isinstance(value, (xr.DataArray, Quantity)):
        return convert_units(value, unit)
    else:
        raise NotImplementedError(f"No implementation for 'convert_to_default_units' with an array of type {type(value)} ({value})")
