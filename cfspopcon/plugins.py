"""Registration of external plugin packages."""

from __future__ import annotations

import importlib
from importlib.resources import files

from .algorithm_class import Algorithm
from .unit_handling import Unit
from .unit_handling.default_units import default_units_map, extend_default_units_map, read_default_units_from_file, reset_default_units

#: A plugin may ship a file of this name in its package root, declaring the default units of the
#: variables it introduces. The shape is that of cfspopcon's own ``variables.yaml``: each variable
#: name maps to a dictionary with a ``default_units`` entry.
PLUGIN_VARIABLES_FILE = "plugin_variables.yaml"


def register_plugins(*package_names: str) -> list[str]:
    """Register plugin packages, and report the algorithms they added.

    A plugin is an importable package whose modules register algorithms; importing the package is
    what registers them, so its ``__init__`` must import the modules which define them. If it ships
    a :data:`PLUGIN_VARIABLES_FILE` in its package root, the default units declared there are loaded
    too, so the plugin needs no units code of its own.

    Registration is all-or-nothing: if any package fails to import, or redefines the default units
    of a variable which is already defined, every algorithm and variable the set added is removed
    again. ``ureg.define`` calls are not undone -- pint has no un-define -- but they are inert unless
    a later plugin names one.

    Args:
        package_names: import names of the plugin packages, e.g. "my_popcon_plugin". The import name
            (underscores) may differ from the distribution name (often hyphenated).

    Returns:
        The names of the algorithms registered, in registration order. A package which is already
        imported registers nothing, so registering it a second time returns an empty list.

    Raises:
        ValueError: if a plugin redefined the default units of an existing variable.
    """
    algorithms, units = dict(Algorithm.instances), default_units_map()
    try:
        for name in package_names:
            importlib.import_module(name)
            _load_plugin_variables(name)

        now = default_units_map()
        redefined = [key for key, old in units.items() if key not in now or not _same_units(now[key], old)]
        if redefined:
            raise ValueError(
                f"Plugin(s) {', '.join(repr(name) for name in package_names)} redefine the default units of "
                f"[{', '.join(redefined)}] (rolled back)."
            )
    except BaseException:
        Algorithm.instances.clear()
        Algorithm.instances.update(algorithms)
        reset_default_units()
        extend_default_units_map(units)
        raise

    return [name for name in Algorithm.instances if name not in algorithms]


def _same_units(new: str | None, old: str | None) -> bool:
    """Whether two default_units entries mean the same thing, e.g. "m**3" and "meter ** 3"."""
    if new == old:
        return True
    return new is not None and old is not None and Unit(new) == Unit(old)


def _load_plugin_variables(package_name: str) -> None:
    """Load the package's own variables file, if it ships one."""
    variables_file = files(package_name).joinpath(PLUGIN_VARIABLES_FILE)
    if variables_file.is_file():
        read_default_units_from_file(variables_file)
