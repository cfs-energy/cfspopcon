"""Physics calculations & lumped-parameter models."""
from importlib.metadata import metadata

__version__ = metadata(__package__)["Version"]
__author__ = metadata(__package__)["Author"]

from . import algorithms, file_io, formulas, helpers, named_options, unit_handling
from .algorithm_class import Algorithm, CompositeAlgorithm
from .input_file_handling import read_case
from .plotting import read_plot_style
from .point_selection import find_coords_of_maximum, find_coords_of_minimum
from .read_atomic_data import AtomicData
from .unit_handling import (
    Quantity,
    Unit,
    convert_to_default_units,
    convert_units,
    default_unit,
    magnitude,
    magnitude_in_default_units,
    set_default_units,
    ureg,
    wraps_ufunc,
)

Algorithm.write_yaml()

# export main classes users should need as well as the option enums
__all__ = [
    "helpers",
    "named_options",
    "formulas",
    "unit_handling",
    "ureg",
    "Quantity",
    "Unit",
    "wraps_ufunc",
    "magnitude_in_default_units",
    "set_default_units",
    "default_unit",
    "convert_to_default_units",
    "convert_units",
    "magnitude",
    "read_case",
    "read_plot_style",
    "find_coords_of_maximum",
    "find_coords_of_minimum",
    "Algorithm",
    "CompositeAlgorithm",
    "file_io",
    "AtomicData",
    "algorithms",
]
