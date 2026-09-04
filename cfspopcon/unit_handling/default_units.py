"""Define default units for writing to/from disk."""

from collections.abc import Iterable
from importlib.resources import files
from importlib.resources.abc import Traversable
from numbers import Number
from pathlib import Path
from typing import Any, overload

import numpy as np
import xarray as xr
import yaml
from pint import DimensionalityError, UndefinedUnitError

from .setup_unit_handling import Quantity, Unit, convert_units, magnitude_in_units


def check_units_are_valid(units_dictionary: dict[str, str | None]) -> None:
    """Ensure every unit in the given mapping is recognized.

    Args:
        units_dictionary: maps each variable name to the unit its values are normalized to,
            with None for a variable which is not a unitful quantity.

    Raises:
        ValueError: if any unit is not recognized, listing the offending entries.
    """
    invalid_units = []
    for key, units in units_dictionary.items():
        try:
            Quantity(1.0, units)
        except UndefinedUnitError:
            invalid_units.append((key, units))

    if invalid_units:
        msg = "The following units are not recognized:\n"
        msg += "\n".join([f"{key}: {units}" for key, units in invalid_units])
        raise ValueError(msg)


def _merge_default_units(units_dictionary: dict[str, str | None]) -> None:
    """Add entries to the default units map.

    Args:
        units_dictionary: maps each variable name to the unit its values are normalized to,
            with None for a variable which is not a unitful quantity.

    Raises:
        ValueError: if a unit is not recognized, or an existing variable's units would change.
    """
    check_units_are_valid(units_dictionary)

    changed = {
        key: new
        for key, new in units_dictionary.items()
        if key in _DEFAULT_UNIT_BY_VARIABLE and not _same_units(new, _DEFAULT_UNIT_BY_VARIABLE[key])
    }
    if changed:
        listed = "\n".join(f"{key}: {_DEFAULT_UNIT_BY_VARIABLE[key]!r} -> {new!r}" for key, new in changed.items())
        raise ValueError(f"Refusing to change the default units of already-defined variables:\n{listed}")

    # Keep the first spelling, so a re-declaration cannot reword an existing entry.
    _DEFAULT_UNIT_BY_VARIABLE.update({key: new for key, new in units_dictionary.items() if key not in _DEFAULT_UNIT_BY_VARIABLE})


def _same_units(new: str | None, old: str | None) -> bool:
    """Whether two spellings name the same unit, e.g. "m**3" and "meter ** 3"."""
    return new == old or (new is not None and old is not None and Unit(new) == Unit(old))


def read_default_units_from_file(units_file: str | Path | Traversable | None = None) -> None:
    """Read a variables YAML file and add its default units to the default units map.

    Args:
        units_file: a YAML file mapping each variable name to an entry with a ``default_units``
            key, in the shape of cfspopcon's own ``variables.yaml``. Defaults to that file.

    Raises:
        ValueError: if a unit is not recognized, or an existing variable's units would change.
    """
    source = files("cfspopcon").joinpath("variables.yaml") if units_file is None else units_file
    if isinstance(source, str):
        source = Path(source)
    variables_dictionary: dict[str, dict[str, Any]] = yaml.safe_load(source.read_text())
    _merge_default_units({key: value["default_units"] for key, value in variables_dictionary.items()})


# Maps a variable name to the unit its values are normalized to, or None for a variable
# which is not a unitful quantity.
_DEFAULT_UNIT_BY_VARIABLE: dict[str, str | None] = {}
read_default_units_from_file()


def extend_default_units_map(units_dictionary: dict[str, str | None]) -> None:
    """Extend the default units map with the given dictionary.

    Args:
        units_dictionary: maps each variable name to the unit its values are normalized to,
            with None for a variable which is not a unitful quantity.

    Raises:
        ValueError: if a unit is not recognized, or an existing variable's units would change.
    """
    _merge_default_units(units_dictionary)


def default_units_map() -> dict[str, str | None]:
    """Return a copy of the default units map.

    Returns:
        Maps each variable name to the unit its values are normalized to, with None for a
        variable which is not a unitful quantity.
    """
    return dict(_DEFAULT_UNIT_BY_VARIABLE)


def reset_default_units() -> None:
    """Reset the default units to an empty dictionary."""
    global _DEFAULT_UNIT_BY_VARIABLE  # noqa: PLW0603
    _DEFAULT_UNIT_BY_VARIABLE = {}


def default_unit(var: str) -> str | None:
    """Return cfspopcon's default unit for a given quantity.

    The mapping is seeded from cfspopcon's own ``variables.yaml`` and extended as further
    plugins are registered. Use :func:`extend_default_units_map` or
    :func:`read_default_units_from_file` to add entries directly.

    Args:
        var: Quantity name

    Returns: Unit
    """
    try:
        return _DEFAULT_UNIT_BY_VARIABLE[var]
    except KeyError:
        raise KeyError(
            f"No default unit defined for {var}. Please check configured default units in the unit_handling submodule."
        ) from None


def magnitude_in_default_units(value: Quantity | xr.DataArray, key: str) -> float | list[float] | Any:
    """Convert values to default units and then return the magnitude.

    Args:
        value: input value to convert to a float
        key: name of variable which we are fetching the default units for

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
    if isinstance(mag, np.ndarray | xr.DataArray):
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
        key: name of variable which we are setting the default units for

    Returns:
        magnitude of value in default units
    """

    def _is_number_not_bool(val: Any) -> bool:
        return isinstance(val, Number) and not isinstance(val, bool)

    def _is_iterable_of_number_not_bool(val: Any) -> bool:
        if not isinstance(val, Iterable):
            return False

        if isinstance(val, np.ndarray | xr.DataArray) and val.ndim == 0:
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


def convert_to_default_units(value: float | Quantity | xr.DataArray, key: str) -> float | Quantity | xr.DataArray:
    """Convert an array or scalar to default units."""
    unit = default_unit(key)
    if unit is None:
        return value
    elif isinstance(value, xr.DataArray | Quantity):
        return convert_units(value, unit)
    else:
        raise NotImplementedError(f"No implementation for 'convert_to_default_units' with an array of type {type(value)} ({value})")
