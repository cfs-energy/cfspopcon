"""Automatic discovery of registered algorithms.

Replaces the "import every submodule in ``__init__.py`` to register" pattern with a ``pkgutil`` walk
of :mod:`cfspopcon.formulas`, plus an entry-point group (``cfspopcon.algorithms``) through which an
installed distribution can contribute algorithms with no cfspopcon-side import.

Discovery is explicit -- importing ``cfspopcon`` registers nothing -- and runs in two phases, so the
order the walk visits modules in does not matter: the walk only registers algorithms and *declares*
composites (see :meth:`~cfspopcon.algorithm_class.CompositeAlgorithm.register_from_list`), which are
built once every component is registered.
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

#: True while a walk is in progress, so a walk nested inside another leaves the composite build to
#: the outermost one, by which time both have registered their algorithms.
_walking = False


def _discover(package: ModuleType | str, load_entry_points: bool = False) -> None:
    """Import every submodule of ``package``, then build whatever composites that satisfies."""
    from .algorithm_class import build_pending_composites

    global _walking  # noqa: PLW0603
    outermost = not _walking
    _walking = True
    try:
        if isinstance(package, str):
            package = importlib.import_module(package)
        for info in pkgutil.walk_packages(package.__path__, prefix=f"{package.__name__}."):
            importlib.import_module(info.name)
        if load_entry_points:
            load_entry_point_algorithms()
    finally:
        _walking = not outermost
    if outermost:
        build_pending_composites()


def discover_algorithms_in_package(package: ModuleType | str) -> None:
    """Register every algorithm defined anywhere beneath ``package``, given as a module or its name.

    Lets a package which builds on cfspopcon register all of its algorithms without importing each
    module by hand. A new subfolder needs an ``__init__.py``: a directory without one is not walked.

    Composites the walk declares are built when it finishes, so they may name anything registered by
    then -- including cfspopcon's own algorithms, which means
    :func:`discover_builtin_algorithms` has to have run first.
    """
    _discover(package)


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

    Composites are built only once both the walk and the entry points have registered their
    algorithms, so a downstream composite may be built from cfspopcon's algorithms and vice versa.
    """
    from . import formulas

    _discover(formulas, load_entry_points=True)
