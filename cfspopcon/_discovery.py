"""Automatic discovery of registered algorithms.

Replaces the "import every submodule in ``__init__.py`` to register" pattern with a ``pkgutil``
walk of :mod:`cfspopcon.formulas` (so adding ``formulas/foo/bar.py`` is enough, though a new subfolder
still needs an ``__init__.py`` -- a directory without one is not walked), plus an entry-point
group (``cfspopcon.algorithms``) through which an installed distribution can contribute algorithms
with no cfspopcon-side import. Both run lazily, once, on the first registry query; the
``@Algorithm.register_algorithm`` decorator is unchanged.

Discovery runs in two phases, so the order the walk visits modules in does not matter: the walk only
registers algorithms and *declares* composites (see
:meth:`~cfspopcon.algorithm_class.CompositeAlgorithm.register_from_list`), which are built
afterwards, once every component is registered.

Discovery is all-or-nothing: anything which goes wrong -- a module that will not import, a broken
entry point, a composite naming an algorithm nobody registers -- fails the query that triggered it,
and every later query re-raises that same error. Half a registry is not worth recovering.
"""

from __future__ import annotations

import importlib
import importlib.util
import pkgutil
from importlib.metadata import entry_points
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType, TracebackType

#: Entry-point group downstream packages declare to contribute algorithms. The target may be a
#: module (imported for its ``@register`` side effects) or a callable taking no arguments (invoked
#: to register explicitly, with no import-time side effects). A target must only *register*:
#: composites are not built until every provider has been loaded, so looking one up from an
#: entry-point target will not find it.
ENTRY_POINT_GROUP = "cfspopcon.algorithms"

_discovered = False
_discovering = False
_failure: BaseException | None = None
_failure_traceback: TracebackType | None = None


def discover_algorithms_in_package(package: ModuleType | str) -> None:
    """Import every submodule of ``package`` so its ``@Algorithm.register_algorithm`` decorators run.

    ``package`` is an imported package or its dotted name. Walking the package registers every
    algorithm defined anywhere beneath it, so a package (cfspopcon or one that builds on it) can
    register all of its algorithms without importing each module by hand.

    cfspopcon's own algorithms are registered first, so a composite declared by ``package`` may be
    built from them. Composites declared with
    :meth:`~cfspopcon.algorithm_class.CompositeAlgorithm.register_from_list` are built once the
    outermost walk finishes, since a registry query from a module being imported re-enters here.

    A walk which raises has left the registry half-populated, so it is latched and re-raised by
    every later query. A composite which could not be built is not: its declaration stays pending,
    and walking the package which supplies the missing algorithm still completes it.
    """
    from .algorithm_class import build_pending_composites

    global _discovering, _failure, _failure_traceback  # noqa: PLW0603

    ensure_discovered()  # whatever the walk imports may refer to cfspopcon's own algorithms

    name = package if isinstance(package, str) else None
    if name is not None and importlib.util.find_spec(name) is None:
        # A name which does not resolve has run no code, so it must not poison the registry.
        raise ModuleNotFoundError(f"No module named {name!r}", name=name)

    outermost = not _discovering
    _discovering = True  # a registry query from a walked module must not build composites yet
    try:
        if name is not None:
            package = importlib.import_module(name)
        for info in pkgutil.walk_packages(package.__path__, prefix=f"{package.__name__}."):  # type:ignore[union-attr]
            importlib.import_module(info.name)
    except BaseException as exc:
        # A half-walked package poisons the registry; say so on every later query. BaseException,
        # so that interrupting a slow first discovery does not leave usable-looking leftovers.
        _failure, _failure_traceback = exc, exc.__traceback__
        raise
    finally:
        _discovering = not outermost

    if outermost:
        build_pending_composites()


def discover_builtin_algorithms() -> None:
    """Populate the registry now rather than on the first query, by walking :mod:`cfspopcon.formulas`.

    Entry-point providers are loaded too, since this runs the same discovery the first query would.
    Reading the registry already does this for you; call it to front-load the cost, to fail early on
    a broken installation, or to pick up a module added since discovery last ran.
    """
    from . import formulas

    discover_algorithms_in_package(formulas)


def load_entry_point_algorithms() -> None:
    """Load algorithm providers declared by any installed distribution via entry points."""
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        obj = ep.load()  # importing the target already runs a module's side effects
        if callable(obj):
            obj()  # a callable target registers explicitly (preferred; no import-time side effects)


def ensure_discovered() -> None:
    """Run built-in + entry-point discovery exactly once (idempotent).

    Composites are built only once both the built-in walk and the entry points have registered
    their algorithms, so a downstream composite may be built from cfspopcon's algorithms and vice
    versa.

    A registry query made from a module being imported by the walk re-enters this function; that
    re-entrant call returns immediately rather than restarting the walk. A failure is remembered
    and re-raised, so a half-discovered registry is never handed back as if it were complete.
    """
    from .algorithm_class import build_pending_composites

    global _discovered, _discovering, _failure, _failure_traceback  # noqa: PLW0603
    if _failure is not None:
        # Restore the traceback it failed with: re-raising the object as-is would grow its
        # traceback by a frame on every registry query for the rest of the process.
        raise _failure.with_traceback(_failure_traceback)
    if _discovered or _discovering:
        return
    _discovering = True
    try:
        discover_builtin_algorithms()
        load_entry_point_algorithms()
        build_pending_composites()
    except BaseException as exc:
        _failure, _failure_traceback = exc, exc.__traceback__
        raise
    finally:
        _discovering = False
    _discovered = True
