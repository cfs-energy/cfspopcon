"""Automatic discovery of registered algorithms.

To use cfspopcon's algorithms, all that is needed is::

    import cfspopcon

    cfspopcon.discover_builtin_algorithms()

That registers every algorithm cfspopcon defines and builds every composite, after which they can be
looked up by name in ``cfspopcon.registry``. Importing cfspopcon on its own registers nothing.

The algorithms are found by walking :mod:`cfspopcon.formulas` with ``pkgutil``, rather than by
importing each submodule by hand in an ``__init__.py``. A separate package can add algorithms of its
own, either by listing itself under :data:`ENTRY_POINT_GROUP` or by calling
:func:`discover_algorithms_in_package`.

Discovery runs in two phases, so the order the walk happens to visit modules in does not matter. The
walk registers algorithms, but only *declares* composites (see
:meth:`~cfspopcon.algorithm_class.CompositeAlgorithm.register_from_list`). The declared composites are
built afterwards, by which point everything they are built from has been registered.
"""

from __future__ import annotations

import importlib
import pkgutil
from importlib.metadata import entry_points
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

#: The name of the entry-point group cfspopcon looks in for algorithms defined outside it. A package
#: which lists itself under this group in its own packaging metadata is loaded during discovery::
#:
#:     [project.entry-points."cfspopcon.algorithms"]
#:     my_package = "my_package:register"
#:
#: This is how the ``popcon`` command-line interface reaches algorithms it does not itself ship: it
#: is never told which packages to import, so an entry point is the only thing that tells it. A
#: constant rather than an argument, because a downstream package has to spell the same string in its
#: metadata for any of this to work; there is no use in reading a different group.
ENTRY_POINT_GROUP = "cfspopcon.algorithms"

#: True while a walk is in progress, so a walk nested inside another leaves the composite build to
#: the outermost one, by which time both have registered their algorithms.
_walking = False

#: Entry-point targets loaded so far, by target specification.
_loaded_entry_points: set[str] = set()


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
    """Load the algorithm providers which installed packages list under :data:`ENTRY_POINT_GROUP`.

    A provider points its entry point at either:

    * a module, which is imported, so that registering happens as a side effect of running it; or
    * a function taking no arguments, which is called, so that nothing is registered merely by
      importing the package.

    Either way the target should only register algorithms, and should not look one up. Composites are
    built after every provider has been loaded, so no composite is in the registry yet at this point.

    A target is loaded once per process, however many times discovery is called. Importing a module
    twice does nothing, since Python caches it, but calling a function twice would try to register its
    algorithms a second time, which is an error. A target counts as loaded only after it returns
    without raising, so one which fails keeps reporting that failure instead of being skipped from
    the next call onwards.
    """
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        if ep.value in _loaded_entry_points:
            continue
        obj = ep.load()  # importing the target already runs a module's side effects
        if callable(obj):
            obj()  # a callable target registers explicitly (preferred; no import-time side effects)
        _loaded_entry_points.add(ep.value)


def discover_builtin_algorithms() -> None:
    """Register every algorithm cfspopcon and its entry-point providers define.

    Call this before using the registry: ``import cfspopcon`` deliberately registers nothing.
    Repeated calls are cheap and change nothing: the modules walked are already imported, and each
    entry-point target is loaded only once.

    Composites are built only once both the walk and the entry points have registered their
    algorithms, so a downstream composite may be built from cfspopcon's algorithms and vice versa.
    """
    from . import formulas

    _discover(formulas, load_entry_points=True)
