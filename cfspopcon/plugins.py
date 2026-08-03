"""Explicit registration of external plugin packages."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files

from .algorithm_class import (
    Algorithm,
    CompositeAlgorithm,
    deferred_composite_build,
    discover_algorithms_in_packages,
    discover_builtin_algorithms,
    pending_composites,
    registered_algorithms,
    restore_registry,
)
from .unit_handling import ureg
from .unit_handling.default_units import default_units_map, extend_default_units_map, read_default_units_from_file, reset_default_units

#: A plugin may ship a file of this name in its package root, declaring the default units of the
#: variables it introduces. The shape is that of cfspopcon's own ``variables.yaml`` — each variable
#: name maps to a dictionary with a ``default_units`` entry; a flat ``name: units`` mapping is not
#: accepted::
#:
#:     widget_temperature:
#:       default_units: kelvin
#:       description: Operating temperature of the widget.
#:
#: A ``description`` is encouraged but not yet surfaced anywhere; ``set_by``/``used_by`` are ignored.
PLUGIN_VARIABLES_FILE = "plugin_variables.yaml"


@dataclass(frozen=True)
class PluginReport:
    """A summary of the algorithms, variables and units a plugin registered."""

    plugin: str
    algorithms: tuple[str, ...] = ()
    variables: tuple[str, ...] = ()
    units: tuple[str, ...] = ()

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

#: Marks a variable whose default units a plugin removed, which None cannot: None is a legitimate
#: value, meaning "not a unitful quantity".
_REMOVED = object()


@dataclass
class _Snapshot:
    """Everything a plugin can add to, captured so it can be put back."""

    algorithms: dict[str, Algorithm | CompositeAlgorithm]
    pending: list[tuple[str, list[str]]]
    units: dict[str, str | None]
    pint_units: set[str]

    @classmethod
    def take(cls) -> _Snapshot:
        """Capture the current state of every registry a plugin writes to."""
        return cls(registered_algorithms(), pending_composites(), default_units_map(), set(iter(ureg)))

    def restore(self) -> None:
        """Put the registries back as they were.

        Unit *definitions* made with ``ureg.define`` are not undone -- pint has no un-define -- so a
        rejected plugin's unit definitions linger. They are inert unless a later plugin names one.
        """
        restore_registry(self.algorithms, self.pending)
        reset_default_units()
        extend_default_units_map(self.units)


def _report(plugin: str, start: _Snapshot, end: _Snapshot) -> PluginReport:
    """Describe what one plugin's walk added, by diffing the snapshots taken either side of it.

    Composites the plugin declared are counted as its algorithms even though they are built later,
    once every plugin has been walked: a report is only ever returned if that build succeeded, so by
    then every declaration has become a registered algorithm.
    """
    declared = {name for name, _ in end.pending} - {name for name, _ in start.pending}
    return PluginReport(
        plugin=plugin,
        algorithms=tuple(sorted((set(end.algorithms) - set(start.algorithms)) | declared)),
        variables=tuple(sorted(set(end.units) - set(start.units))),
        units=tuple(sorted(end.pint_units - start.pint_units)),
    )


def _clashes(before: _Snapshot, after: _Snapshot) -> list[str]:
    """Describe everything the plugins redefined, rather than added."""
    clashes = []
    for key, old in before.units.items():
        new = after.units.get(key, _REMOVED)
        if new != old:
            shown = "(removed)" if new is _REMOVED else repr(new)
            clashes.append(f"variable '{key}' default units: {old!r} -> {shown}")
    # Identity, not membership: an algorithm registered with override=True keeps its name, so
    # comparing name sets alone would report a replaced builtin as an untouched one.
    clashes += [
        f"algorithm '{name}' was replaced" for name, algorithm in before.algorithms.items() if after.algorithms.get(name) is not algorithm
    ]
    return clashes


def _offender_message(offenders: dict[str, list[str]]) -> str:
    """Report every plugin which redefined something, and what."""
    listed = "\n".join(f"  {plugin}:\n" + "\n".join(f"    {clash}" for clash in clashes) for plugin, clashes in offenders.items())
    return f"Plugin(s) {', '.join(repr(name) for name in offenders)} redefine what is already defined (rolled back):\n{listed}"


def _cache_rejection(fresh: list[str], offenders: dict[str, list[str]]) -> None:
    """Record why each plugin in a rejected set was rolled back.

    Only clashes are cached: they are deterministic, so a repeat would reach the same verdict, and a
    repeat cannot reach it honestly -- the modules are still in ``sys.modules`` with their side effects
    undone, so re-walking registers nothing and would look like a success with an empty report.

    A plugin which did not itself clash gets a message saying so, rather than the offender's, since
    otherwise it stands accused of a redefinition it never made.
    """
    if not offenders:
        return

    for name in fresh:
        if name in offenders:
            _failed_registrations[name] = PluginClashError(_offender_message({name: offenders[name]}))
        else:
            _failed_registrations[name] = PluginClashError(
                f"Plugin '{name}' registered nothing: it was rolled back as part of a set in which "
                f"{', '.join(repr(offender) for offender in offenders)} redefined what is already defined. "
                "Register it without the offending plugin(s), in a new process -- its modules are already "
                "imported in this one, so registering it again here would report an empty success."
            )


def _load_plugin_variables(package_name: str) -> None:
    """Load the package's own variables file, if it ships one.

    Loaded *before* the package is walked, so that a plugin module's own
    ``extend_default_units_map`` calls take precedence over the file. That
    matches cfspopcon itself, where ``variables.yaml`` is read when the units map is created and inline
    calls come afterwards.
    """
    variables_file = files(package_name).joinpath(PLUGIN_VARIABLES_FILE)
    if variables_file.is_file():
        read_default_units_from_file(variables_file)


def register_plugins(*package_names: str) -> list[PluginReport]:
    """Register several plugin packages as a single unit, and report what each one added.

    Each package is walked, as :func:`~cfspopcon.algorithm_class.discover_algorithms_in_package` walks one, so a
    plugin may be a package with its algorithms spread over submodules. All of the walks share one
    :func:`~cfspopcon.algorithm_class.deferred_composite_build` block, so a composite declared in one
    plugin may name an algorithm from another, in either direction.

    cfspopcon's own algorithms are discovered first, so that a plugin may name them, and so that a
    plugin algorithm colliding with a builtin is reported here rather than by some later walk.

    A plugin which ships a ``plugin_variables.yaml`` in its package root has it loaded automatically,
    so it need not call ``extend_default_units_map`` itself. cfspopcon's
    own ``variables.yaml`` is never written to; a plugin's variables live only in the plugin.

    Registration is all-or-nothing: if any package fails to import, or any of them redefines
    something cfspopcon already defines, the whole set is rolled back and nothing stays registered. A
    package already registered successfully is not walked again and its original report is returned;
    one already rejected re-raises the original error.

    Args:
        package_names: importable names of the plugin packages, e.g. "my_popcon_plugin". This is the
            import name (underscores), which may differ from the distribution name (often hyphenated).

    Returns:
        One report per name given, in the order given.

    Raises:
        PluginClashError: if a plugin redefined an existing variable's default units or replaced a
            registered algorithm. The whole set is rolled back before raising, and each plugin in it
            records why -- the offenders what they redefined, the rest that they were rolled back
            alongside an offender.
    """
    for name in package_names:
        if name in _failed_registrations:
            raise _failed_registrations[name]

    # A plugin may legitimately discover the builtins itself; doing it here first keeps them out of
    # the snapshot diff, so a rollback cannot delete them.
    discover_builtin_algorithms()

    fresh = [name for name in package_names if name not in _successful_registrations]
    before = _Snapshot.take()
    reports: dict[str, PluginReport] = {}
    offenders: dict[str, list[str]] = {}

    try:
        # One block for the whole set, so each plugin's own walk defers the composite build to the
        # end. Each plugin is diffed against the state it inherited, so a clash is attributed to the
        # plugin that caused it -- including one plugin redefining an earlier plugin's variable.
        with deferred_composite_build():
            for name in fresh:
                start = _Snapshot.take()
                _load_plugin_variables(name)
                discover_algorithms_in_packages(name)
                end = _Snapshot.take()
                reports[name] = _report(name, start, end)
                if clashes := _clashes(start, end):
                    offenders[name] = clashes

        if offenders:
            raise PluginClashError(_offender_message(offenders))
    except BaseException:
        before.restore()
        _cache_rejection(fresh, offenders)
        raise

    _successful_registrations.update(reports)
    return [_successful_registrations[name] for name in package_names]


def register_plugin(package_name: str) -> PluginReport:
    """Register a single plugin package and report what it added.

    Single-plugin spelling of :func:`register_plugins`. Two plugins whose composites name each other
    have to be registered together, in one :func:`register_plugins` call.
    """
    return register_plugins(package_name)[0]
