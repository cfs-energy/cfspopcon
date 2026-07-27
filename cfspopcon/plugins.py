"""Explicit registration of external plugin modules."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field

from .algorithm_class import Algorithm
from .unit_handling import ureg
from .unit_handling.default_units import default_units_map, extend_default_units_map, reset_default_units


@dataclass(frozen=True)
class PluginReport:
    """A summary of the algorithms, variables and units a plugin registered."""

    plugin: str
    algorithms: list[str] = field(default_factory=list)
    variables: list[str] = field(default_factory=list)
    units: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        """Return a human-readable summary of the registered items."""
        return (
            f"Plugin '{self.plugin}' registered:\n"
            f"  algorithms: {', '.join(self.algorithms) or '(none)'}\n"
            f"  variables:  {', '.join(self.variables) or '(none)'}\n"
            f"  units:      {', '.join(self.units) or '(none)'}"
        )


class PluginClashError(RuntimeError):
    """Raised when a plugin redefines something cfspopcon already defines."""


_successful_registrations: dict[str, PluginReport] = {}
_failed_registrations: dict[str, PluginClashError] = {}


def _restore_default_units(snapshot: dict[str, str]) -> None:
    """Replace the default units map with the given snapshot."""
    reset_default_units()
    extend_default_units_map(snapshot)


def _remove_algorithms(names: set[str]) -> None:
    """Remove the given algorithm names from the Algorithm registry."""
    for name in names:
        Algorithm.instances.pop(name, None)


def register_plugin(module_name: str) -> PluginReport:
    """Import a plugin module and validate what it registers.

    A plugin module is an ordinary Python module which registers algorithms,
    variables and units at import time, using the same interfaces as cfspopcon's
    built-in formula modules: the `Algorithm.register_algorithm` decorator,
    `extend_default_units_map` and `ureg.define`. This function imports the
    module, checks that it only adds to the registries (never redefines existing
    entries), and reports what was added.

    Repeated calls with the same module name return the original report, or
    re-raise the original error, without importing again. A module that was
    already imported by other means registers nothing new during this call, so
    its report is empty.

    Example::

        report = cfspopcon.register_plugin("my_popcon_plugin")
        print(report)

    Args:
        module_name: Importable name of the plugin module, e.g. "my_popcon_plugin".
            This is the module's import name (underscores), which may differ from
            the name of the distribution that provides it (often hyphenated).

    Returns:
        A summary of the algorithms, variables and units the plugin registered.

    Raises:
        PluginClashError: If the plugin changed the default units of an existing
            variable. The plugin's registrations are rolled back before raising.
    """
    if module_name in _successful_registrations:
        return _successful_registrations[module_name]
    if module_name in _failed_registrations:
        raise _failed_registrations[module_name]

    units_before = default_units_map()
    algorithms_before = set(Algorithm.instances)
    pint_units_before = set(iter(ureg))

    try:
        importlib.import_module(module_name)
    except Exception:
        # A plugin that fails mid-import must leave no partial registrations behind.
        _remove_algorithms(set(Algorithm.instances) - algorithms_before)
        _restore_default_units(units_before)
        raise

    units_after = default_units_map()
    clashes = {key: (units_before[key], units_after[key]) for key in units_before if units_after[key] != units_before[key]}
    if clashes:
        _remove_algorithms(set(Algorithm.instances) - algorithms_before)
        _restore_default_units(units_before)
        details = "\n".join(f"  {key}: '{old}' -> '{new}'" for key, (old, new) in clashes.items())
        error = PluginClashError(f"Plugin '{module_name}' redefines the default units of existing variables (rolled back):\n{details}")
        # The module stays cached in sys.modules with its side effects rolled back, so a
        # repeated import would register nothing and look like a success. Cache the error
        # to keep repeated calls consistent.
        _failed_registrations[module_name] = error
        raise error

    report = PluginReport(
        plugin=module_name,
        algorithms=sorted(set(Algorithm.instances) - algorithms_before),
        variables=sorted(set(units_after) - set(units_before)),
        units=sorted(set(iter(ureg)) - pint_units_before),
    )
    _successful_registrations[module_name] = report
    return report
