"""Automatic discovery of registered algorithms.

To use cfspopcon's algorithms, all that is needed is::

    import cfspopcon

    cfspopcon.discover_builtin_algorithms()

That registers every algorithm cfspopcon defines and builds every composite, after which they can be
looked up by name in ``cfspopcon.registry``. Importing cfspopcon on its own registers nothing.

The algorithms are found by walking :mod:`cfspopcon.formulas` with ``pkgutil``, rather than by
importing each submodule by hand in an ``__init__.py``. A package built on cfspopcon adds algorithms
of its own the same way, by calling :func:`discover_algorithms_in_package` on itself, so its command
line does both::

    cfspopcon.discover_builtin_algorithms()
    cfspopcon.discover_algorithms_in_package("my_package")

Discovery runs in two phases, so the order the walk happens to visit modules in does not matter. The
walk registers algorithms, but only *declares* composites (see
:meth:`~cfspopcon.algorithm_class.CompositeAlgorithm.register_from_list`). The declared composites are
built afterwards, by which point everything they are built from has been registered.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

#: True while a walk is in progress, so a walk nested inside another leaves the composite build to
#: the outermost one, by which time both have registered their algorithms.
_walking = False


def discover_algorithms_in_package(package: ModuleType | str) -> None:
    """Register every algorithm defined anywhere beneath ``package``, given as a module or its name.

    Lets a package which builds on cfspopcon register all of its algorithms without importing each
    module by hand. A new subfolder needs an ``__init__.py``: a directory without one is not walked.

    Composites the walk declares are built when it finishes, so they may name anything registered by
    then -- including cfspopcon's own algorithms, which means :func:`discover_builtin_algorithms` has
    to have run first. A walk nested inside another leaves the build to the outermost one, so two
    packages may declare composites spanning each other.
    """
    from .algorithm_class import build_pending_composites

    global _walking  # noqa: PLW0603
    outermost = not _walking
    _walking = True
    try:
        if isinstance(package, str):
            package = importlib.import_module(package)
        for info in pkgutil.walk_packages(package.__path__, prefix=f"{package.__name__}."):
            importlib.import_module(info.name)
    finally:
        _walking = not outermost
    if outermost:
        build_pending_composites()


def discover_builtin_algorithms() -> None:
    """Register every algorithm cfspopcon defines, by walking :mod:`cfspopcon.formulas`.

    Call this before using the registry: ``import cfspopcon`` deliberately registers nothing.
    Repeated calls are cheap and change nothing, since the modules walked are already imported.
    """
    from . import formulas

    discover_algorithms_in_package(formulas)
