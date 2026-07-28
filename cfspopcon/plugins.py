"""Explicit registration of external plugin packages."""

from __future__ import annotations

from dataclasses import dataclass

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
from .unit_handling.default_units import default_units_map, extend_default_units_map, reset_default_units


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


def register_plugins(*package_names: str) -> list[PluginReport]:
    """Register several plugin packages as a single unit, and report what each one added.

    Each package is walked, as :func:`~cfspopcon.algorithm_class.discover_algorithms_in_package` walks one, so a
    plugin may be a package with its algorithms spread over submodules. All of the walks share one
    :func:`~cfspopcon.algorithm_class.deferred_composite_build` block, so a composite declared in one
    plugin may name an algorithm from another, in either direction.

    cfspopcon's own algorithms are discovered first, so that a plugin may name them, and so that a
    plugin algorithm colliding with a builtin is reported here rather than by some later walk.

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
            registered algorithm. The whole set is rolled back before raising.
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

    try:
        # One block for the whole set, so each plugin's own walk defers the composite build to the
        # end. The per-plugin diffs are for attribution only; rollback is all-or-nothing.
        with deferred_composite_build():
            for name in fresh:
                start = _Snapshot.take()
                discover_algorithms_in_packages(name)
                reports[name] = _report(name, start, _Snapshot.take())

        clashes = _clashes(before, _Snapshot.take())
        if clashes:
            listed = "\n".join(f"  {clash}" for clash in clashes)
            plugins = ", ".join(f"'{name}'" for name in fresh)
            raise PluginClashError(f"Plugin(s) {plugins} redefine what cfspopcon already defines (rolled back):\n{listed}")
    except BaseException as error:
        before.restore()
        # A rejected plugin stays in sys.modules with its side effects undone, so a repeated call
        # would register nothing and look like a success. Cache the error to keep repeats honest.
        if isinstance(error, PluginClashError):
            _failed_registrations.update(dict.fromkeys(fresh, error))
        raise

    _successful_registrations.update(reports)
    return [_successful_registrations[name] for name in package_names]


def register_plugin(package_name: str) -> PluginReport:
    """Register a single plugin package and report what it added.

    Single-plugin spelling of :func:`register_plugins`. Two plugins whose composites name each other
    have to be registered together, in one :func:`register_plugins` call.
    """
    return register_plugins(package_name)[0]
