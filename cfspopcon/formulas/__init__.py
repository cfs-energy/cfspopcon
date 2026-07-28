"""Formulas used for the POPCON analysis.

Submodules are not imported by hand here: their algorithms are found by
:func:`cfspopcon.algorithm_class.discover_builtin_algorithms`, a ``pkgutil`` walk of this package.
Adding a new ``formulas/...`` module is therefore sufficient to register its algorithms.

Since that walk only runs when discovery is called, submodules are imported on first attribute
access instead, so that reading ``cfspopcon.formulas.geometry`` still works without it.
"""

import importlib
import pkgutil
from types import ModuleType

#: Discovered rather than hand-maintained, but still spelled __all__ so that
#: ``from cfspopcon.formulas import *`` exports the submodules and nothing else.
__all__ = sorted(info.name for info in pkgutil.iter_modules(__path__))  # noqa: PLE0605


def __getattr__(name: str) -> ModuleType:
    """Import a submodule of ``cfspopcon.formulas`` on first attribute access (PEP 562)."""
    if name in __all__:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """List the submodules alongside the usual module attributes, without importing them all."""
    return sorted(set(__all__) | set(globals()))
