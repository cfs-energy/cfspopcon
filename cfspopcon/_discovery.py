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

The walk visits modules in ``pkgutil`` order (alphabetical within each package), and a re-entrant
registry query made while the walk is in progress sees only the algorithms registered so far. A
module that builds a :class:`~cfspopcon.algorithm_class.CompositeAlgorithm` at import time therefore
has to be visited after the modules registering that composite's components; otherwise the lookup
raises ``KeyError``. Build such a composite inside the module that defines its last component, or
in a module sorting after it.
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


def discover_algorithms_in_package(package: ModuleType | str) -> None:
    """Import every submodule of ``package`` so its ``@Algorithm.register_algorithm`` decorators run.

    ``package`` is an imported package or its dotted name. Walking the package registers every
    algorithm defined anywhere beneath it, so a package (cfspopcon or one that builds on it) can
    register all of its algorithms without importing each module by hand.
    """
    if isinstance(package, str):
        package = importlib.import_module(package)

    for info in pkgutil.walk_packages(package.__path__, prefix=f"{package.__name__}."):
        importlib.import_module(info.name)


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

    A registry query made from a module being imported by the walk re-enters this function; that
    re-entrant call returns immediately rather than restarting the walk. A walk which raises leaves
    the flag clear, so the next query retries instead of being stuck with a half-filled registry.
    """
    global _discovered, _discovering  # noqa: PLW0603
    if _discovered or _discovering:
        return
    _discovering = True
    try:
        discover_builtin_algorithms()
        load_entry_point_algorithms()
    finally:
        _discovering = False
    _discovered = True
