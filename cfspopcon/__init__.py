"""Physics calculations & lumped-parameter models."""

from importlib.metadata import metadata

__version__ = metadata(__package__)["Version"]
__author__ = metadata(__package__)["Author"]

from . import file_io, formulas, named_options, shaping_and_selection
from .algorithm_class import (
    Algorithm,
    CompositeAlgorithm,
    discover_algorithms_in_package,
    discover_algorithms_in_packages,
    discover_builtin_algorithms,
    registry,
)
from .deprecation_handler import handle_deprecated_arguments
from .input_file_handling import process_input_dictionary, read_case
from .plotting import read_plot_style
from .plugins import PluginClashError, PluginReport, register_plugin, register_plugins
from .unit_handling import (
    convert_to_default_units,
    convert_units,
    magnitude_in_default_units,
    set_default_units,
)

# export main classes users should need as well as the option enums
__all__ = [
    "Algorithm",
    "CompositeAlgorithm",
    "PluginClashError",
    "PluginReport",
    "convert_to_default_units",
    "convert_units",
    "discover_algorithms_in_package",
    "discover_algorithms_in_packages",
    "discover_builtin_algorithms",
    "file_io",
    "formulas",
    "handle_deprecated_arguments",
    "magnitude_in_default_units",
    "named_options",
    "process_input_dictionary",
    "read_case",
    "read_plot_style",
    "register_plugin",
    "register_plugins",
    "registry",
    "set_default_units",
    "shaping_and_selection",
]
