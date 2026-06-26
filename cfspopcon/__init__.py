"""Physics calculations & lumped-parameter models."""

from importlib.metadata import metadata

__version__ = metadata(__package__)["Version"]
__author__ = metadata(__package__)["Author"]

from . import file_io, formulas, named_options, shaping_and_selection
from ._discovery import discover_algorithms_in_package, ensure_discovered
from .algorithm_class import Algorithm, CompositeAlgorithm, algorithms
from .formulas.atomic_data import AtomicData
from .input_file_handling import process_input_dictionary, read_case
from .plotting import read_plot_style
from .unit_handling import (
    convert_to_default_units,
    convert_units,
    magnitude_in_default_units,
    set_default_units,
)

# Populate the algorithm registry eagerly so `import cfspopcon` behaves as before (algorithms are
# also discovered lazily on first registry query, so direct-submodule use works without this).
ensure_discovered()

# export main classes users should need as well as the option enums
__all__ = [
    "Algorithm",
    "AtomicData",
    "CompositeAlgorithm",
    "algorithms",
    "convert_to_default_units",
    "convert_units",
    "discover_algorithms_in_package",
    "file_io",
    "formulas",
    "magnitude_in_default_units",
    "named_options",
    "process_input_dictionary",
    "read_case",
    "read_plot_style",
    "set_default_units",
    "shaping_and_selection",
]
