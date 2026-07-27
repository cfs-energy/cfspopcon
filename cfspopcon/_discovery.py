"""Automatic discovery of registered algorithms.

Replaces the "import every submodule in ``__init__.py`` to register" pattern with:

* **auto-discovery of cfspopcon's own algorithms** by walking the :mod:`cfspopcon.formulas`
  package (so adding ``formulas/foo/bar.py`` is enough — no hand-maintained import list, no
  "forgot to import it -> silently missing" failure mode), and
* **discovery of downstream-provided algorithms via entry points** (group
  ``cfspopcon.algorithms``), so an installed distribution can contribute algorithms without any
  cfspopcon-side import.

Both run lazily and exactly once, the first time the registry is queried (see
:meth:`cfspopcon.algorithm_class.Algorithm.algorithms`). The ``@Algorithm.register_algorithm``
decorator is unchanged; discovery only automates *which modules get imported*.

Discovery runs in two phases, so the order the walk happens to visit modules in does not matter.
The walk itself only registers algorithms and *declares* composites (see
:meth:`~cfspopcon.algorithm_class.CompositeAlgorithm.register_from_list`); the declarations are
built afterwards, once every component is registered. Because a composite may be built from other
composites, that build repeats until all declarations are satisfied.
"""

from __future__ import annotations

import importlib
import pkgutil
from importlib.metadata import entry_points
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

#: Entry-point group downstream packages declare to contribute algorithms. The target may be a
#: module (imported for its ``@register`` side effects) or a callable (invoked to register
#: explicitly, with no import-time side effects).
ENTRY_POINT_GROUP = "cfspopcon.algorithms"

_discovered = False
_discovering = False


def discover_algorithms_in_package(package: ModuleType | str, build_composites: bool = True) -> None:
    """Import every submodule of ``package`` so its ``@Algorithm.register_algorithm`` decorators run.

    ``package`` is an imported package or its dotted name. Walking the package registers every
    algorithm defined anywhere beneath it, so a package (cfspopcon or one that builds on it) can
    register all of its algorithms without importing each module by hand.

    Composites declared with :meth:`~cfspopcon.algorithm_class.CompositeAlgorithm.register_from_list`
    are built afterwards. Pass ``build_composites=False`` to leave them pending, when a later
    discovery step still has to register the algorithms they are built from.
    """
    from .algorithm_class import Algorithm, _pending_composites, build_pending_composites

    if isinstance(package, str):
        package = importlib.import_module(package)

    for info in pkgutil.walk_packages(package.__path__, prefix=f"{package.__name__}."):
        # Python drops a module which raises partway from sys.modules, so a later attempt at
        # discovery re-runs its body from the top. Undo whatever it registered before it raised,
        # otherwise that re-run collides with its own leftovers and discovery can never succeed.
        registered = set(Algorithm.instances)
        declared = len(_pending_composites)
        try:
            importlib.import_module(info.name)
        except BaseException:
            for key in set(Algorithm.instances) - registered:
                del Algorithm.instances[key]
            del _pending_composites[declared:]
            raise

    if build_composites:
        build_pending_composites()


def discover_builtin_algorithms(build_composites: bool = True) -> None:
    """Register cfspopcon's own algorithms by walking the :mod:`cfspopcon.formulas` package."""
    from . import formulas

    discover_algorithms_in_package(formulas, build_composites=build_composites)


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
    re-entrant call returns immediately rather than restarting the walk. A walk which raises leaves
    the flag clear, so the next query retries instead of being stuck with a half-filled registry.
    """
    from .algorithm_class import build_pending_composites

    global _discovered, _discovering  # noqa: PLW0603
    if _discovered or _discovering:
        return
    _discovering = True
    try:
        discover_builtin_algorithms(build_composites=False)
        load_entry_point_algorithms()
        build_pending_composites()
    finally:
        _discovering = False
    _discovered = True
