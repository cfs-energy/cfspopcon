"""Define default units for writing to/from disk."""

from collections.abc import Iterable
from importlib.resources import as_file, files
from numbers import Number
from pathlib import Path
from typing import Any, Optional, Union, overload
from warnings import warn

import numpy as np
import xarray as xr
import yaml
from pint import DimensionalityError, UndefinedUnitError

from .setup_unit_handling import Quantity, convert_units, magnitude_in_units

# Module global stat holding the registered default units mapping
_DEFAULT_UNITS: dict[str, str] = {}


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


def read_default_units_from_file(filepath: Optional[Path] = None) -> None:
    """Read in a units YAML ifile and add the units to the registered default units map.

    Args:
        filepath: yaml file to read. If none, cfspopcon's default_units.yaml is read.

    """
    if filepath is None:
        with as_file(files("cfspopcon").joinpath("default_units.yaml")) as fp:
            with open(fp) as f:
                units_dictionary: dict[str, str] = yaml.safe_load(f)
    else:
        units_dictionary = yaml.safe_load(filepath.read_text())

    check_units_are_valid(units_dictionary)

    global _DEFAULT_UNITS  # noqa: PLW0603
    _DEFAULT_UNITS |= units_dictionary


def extend_default_units_map(units_dictionary: dict[str, str]) -> None:
    """Extend the default units map with the given dictionary.

    Args:
        units_dictionary: dictionary of units to add to the default units map
    """
    check_units_are_valid(units_dictionary)
    global _DEFAULT_UNITS  # noqa: PLW0603
    _DEFAULT_UNITS |= units_dictionary


def reset_default_units() -> None:
    """Reset the default units to an empty dictionary."""
    global _DEFAULT_UNITS  # noqa: PLW0603
    _DEFAULT_UNITS = {}


def default_unit(var: str) -> Union[str, None]:
    """Return cfspopcon's default unit for a given quantity.

    The mapping of variable name to default unit is loaded upon module import.
    By default this mapping will be initialized by the default_units.yaml file
    in the cfspopcon package. To modify the default units mapping see, use any
    of the following functions:
    - `read_default_units_from_file`
    - `extend_default_units_map`
    - `reset_default_units`

    Args:
        var: Quantity name

    Returns: Unit
    """
    try:
        return _DEFAULT_UNITS[var]
    except KeyError:
        raise KeyError(
            f"No default unit defined for {var}. Please check configured default units in the unit_handling submodule."
        ) from None


def magnitude_in_default_units(value: Union[Quantity, xr.DataArray], key: str) -> Union[float, list[float], Any]:
    """Convert values to default units and then return the magnitude.

    Args:
        value: input value to convert to a float
        key: name of field for looking up default unit

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
        print(f"Unit conversion failed for {key}. Could not convert '{value}' to '{default_unit(key)}'")
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
    unit = default_unit(key)
    if unit is None:
        if _is_number_not_bool(value) or _is_iterable_of_number_not_bool(value):
            raise RuntimeError(
                f"set_default_units for key {key} and value {value} of type {type(value)}: numeric types should carry units!"
            )
        return value
    elif isinstance(value, xr.DataArray):
        return value.pint.quantify(unit)
    else:
        return Quantity(value, unit)


@overload
def convert_to_default_units(value: float, key: str) -> float: ...


@overload
def convert_to_default_units(value: xr.DataArray, key: str) -> xr.DataArray: ...


@overload
def convert_to_default_units(value: Quantity, key: str) -> Quantity: ...


def convert_to_default_units(value: Union[float, Quantity, xr.DataArray], key: str) -> Union[float, Quantity, xr.DataArray]:
    """Convert an array or scalar to default units."""
    unit = default_unit(key)
    if unit is None:
        return value
    elif isinstance(value, (xr.DataArray, Quantity)):
        return convert_units(value, unit)
    else:
        raise NotImplementedError(f"No implementation for 'convert_to_default_units' with an array of type {type(value)} ({value})")
