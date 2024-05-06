"""Physics calculations & lumped-parameter models."""
from importlib.metadata import metadata

__version__ = metadata(__package__)["Version"]
__author__ = metadata(__package__)["Author"]

from . import file_io, formulas, named_options
from .algorithm_class import Algorithm, CompositeAlgorithm
from .formulas.read_atomic_data import AtomicData
from .input_file_handling import read_case
from .plotting import read_plot_style
from .point_selection import find_coords_of_maximum, find_coords_of_minimum
from .unit_handling import (
    convert_to_default_units,
    convert_units,
    magnitude_in_default_units,
    set_default_units,
)

# export main classes users should need as well as the option enums
__all__ = [
    "formulas",
    "named_options",
    "magnitude_in_default_units",
    "convert_to_default_units",
    "set_default_units",
    "convert_units",
    "read_case",
    "read_plot_style",
    "find_coords_of_maximum",
    "find_coords_of_minimum",
    "Algorithm",
    "CompositeAlgorithm",
    "file_io",
    "AtomicData",
]
