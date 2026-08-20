"""Formulas used for the POPCON analysis.

Submodules are not imported here: :func:`cfspopcon.algorithm_class.discover_builtin_algorithms`
walks this package, so adding a ``formulas/...`` module is enough to register its algorithms.
Attribute access imports a submodule on demand, so ``cfspopcon.formulas.geometry`` still resolves
when discovery has not run.
"""

import importlib
import pkgutil
from types import ModuleType

#: Discovered rather than hand-maintained, but spelled __all__ so ``import *`` exports the submodules.
__all__ = sorted(info.name for info in pkgutil.iter_modules(__path__))  # noqa: PLE0605


def __getattr__(name: str) -> ModuleType:
    """Import a submodule of ``cfspopcon.formulas`` on first attribute access (PEP 562)."""
    if name in __all__:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """List the submodules alongside the usual module attributes, without importing them all."""
    return sorted(set(__all__) | set(globals()))
