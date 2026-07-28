"""Automatic discovery of registered algorithms.

Replaces the "import every submodule in ``__init__.py`` to register" pattern with a ``pkgutil``
walk of :mod:`cfspopcon.formulas` (so adding ``formulas/foo/bar.py`` is enough, though a new subfolder
still needs an ``__init__.py`` -- a directory without one is not walked), plus an entry-point
group (``cfspopcon.algorithms``) through which an installed distribution can contribute algorithms
with no cfspopcon-side import. The ``@Algorithm.register_algorithm`` decorator is unchanged.

Discovery is explicit: importing ``cfspopcon`` registers nothing, and
:func:`discover_builtin_algorithms` populates the registry when a caller asks for it. Until then the
registry is empty, so a lookup which fails says that discovery has not run rather than reporting an
algorithm as missing.

Discovery runs in two phases, so the order the walk visits modules in does not matter: the walk only
registers algorithms and *declares* composites (see
:meth:`~cfspopcon.algorithm_class.CompositeAlgorithm.register_from_list`), which are built
afterwards, once every component is registered.
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

#: True while a walk is in progress, so that a walk nested inside another (an entry-point provider
#: which walks its own package) leaves the composite build to the outermost one.
_walking = False


def _walk(package: ModuleType | str) -> None:
    """Import every submodule of ``package``, so its registration decorators run."""
    if isinstance(package, str):
        package = importlib.import_module(package)
    for info in pkgutil.walk_packages(package.__path__, prefix=f"{package.__name__}."):
        importlib.import_module(info.name)


def discover_algorithms_in_package(package: ModuleType | str) -> None:
    """Import every submodule of ``package`` so its ``@Algorithm.register_algorithm`` decorators run.

    ``package`` is an imported package or its dotted name. Walking the package registers every
    algorithm defined anywhere beneath it, so a package which builds on cfspopcon can register all
    of its algorithms without importing each module by hand.

    Composites declared with :meth:`~cfspopcon.algorithm_class.CompositeAlgorithm.register_from_list`
    are built once the outermost walk finishes, so a walk nested inside another may declare
    composites which combine its algorithms with the outer package's. A composite built from one of
    cfspopcon's own algorithms therefore needs :func:`discover_builtin_algorithms` to have run first.
    """
    from .algorithm_class import build_pending_composites

    global _walking  # noqa: PLW0603
    outermost = not _walking
    _walking = True
    try:
        _walk(package)
    finally:
        _walking = not outermost
    if outermost:
        build_pending_composites()


def load_entry_point_algorithms() -> None:
    """Load algorithm providers declared by any installed distribution via entry points."""
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        obj = ep.load()  # importing the target already runs a module's side effects
        if callable(obj):
            obj()  # a callable target registers explicitly (preferred; no import-time side effects)


def discover_builtin_algorithms() -> None:
    """Register every algorithm cfspopcon and its entry-point providers define.

    Call this before using the registry: ``import cfspopcon`` deliberately registers nothing.
    Repeated calls are cheap and change nothing, since the modules walked are already imported.

    Composites are built only once both the built-in walk and the entry points have registered their
    algorithms, so a downstream composite may be built from cfspopcon's algorithms and vice versa.
    """
    from .algorithm_class import build_pending_composites

    global _walking  # noqa: PLW0603
    _walking = True  # a provider's own walk must not build composites before every provider is loaded
    try:
        from . import formulas

        _walk(formulas)
        load_entry_point_algorithms()
    finally:
        _walking = False
    build_pending_composites()
