"""Automatic discovery of registered algorithms.

Replaces the "import every submodule in ``__init__.py`` to register" pattern with a ``pkgutil``
walk of :mod:`cfspopcon.formulas` (so adding ``formulas/foo/bar.py`` is enough), plus an entry-point
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
import pkgutil
from importlib.metadata import entry_points
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

#: Entry-point group downstream packages declare to contribute algorithms. The target may be a
#: module (imported for its ``@register`` side effects) or a callable taking no arguments (invoked
#: to register explicitly, with no import-time side effects). A target must only *register*:
#: composites are not built until every provider has been loaded, so looking one up from an
#: entry-point target will not find it.
ENTRY_POINT_GROUP = "cfspopcon.algorithms"

_discovered = False
_discovering = False
_failure: Exception | None = None


def discover_algorithms_in_package(package: ModuleType | str) -> None:
    """Import every submodule of ``package`` so its ``@Algorithm.register_algorithm`` decorators run.

    ``package`` is an imported package or its dotted name. Walking the package registers every
    algorithm defined anywhere beneath it, so a package (cfspopcon or one that builds on it) can
    register all of its algorithms without importing each module by hand.

    cfspopcon's own algorithms are registered first, so a composite declared by ``package`` may be
    built from them. Composites declared with
    :meth:`~cfspopcon.algorithm_class.CompositeAlgorithm.register_from_list` are built once the
    outermost walk finishes, since a registry query from a module being imported re-enters here.
    """
    from .algorithm_class import build_pending_composites

    global _discovering, _failure  # noqa: PLW0603

    ensure_discovered()  # whatever the walk imports may refer to cfspopcon's own algorithms

    if isinstance(package, str):
        package = importlib.import_module(package)

    outermost = not _discovering
    _discovering = True  # a registry query from a walked module must not build composites yet
    try:
        for info in pkgutil.walk_packages(package.__path__, prefix=f"{package.__name__}."):
            importlib.import_module(info.name)
        if outermost:
            build_pending_composites()
    except Exception as exc:
        _failure = exc  # a half-walked package poisons the registry; say so on every later query
        raise
    finally:
        _discovering = not outermost


def discover_builtin_algorithms() -> None:
    """Register cfspopcon's own algorithms by walking the :mod:`cfspopcon.formulas` package."""
    from . import formulas

    discover_algorithms_in_package(formulas)


def load_entry_point_algorithms(group: str = ENTRY_POINT_GROUP) -> None:
    """Load algorithm providers declared by any installed distribution via entry points."""
    for ep in entry_points(group=group):
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

    global _discovered, _discovering, _failure  # noqa: PLW0603
    if _failure is not None:
        raise _failure
    if _discovered or _discovering:
        return
    _discovering = True
    try:
        discover_builtin_algorithms()
        load_entry_point_algorithms()
        build_pending_composites()
    except Exception as exc:
        _failure = exc
        raise
    finally:
        _discovering = False
    _discovered = True
