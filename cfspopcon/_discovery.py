"""Order-independent discovery of registered algorithms.

Replaces the fragile "import every submodule in ``__init__.py`` to register" pattern with:

* **auto-discovery of cfspopcon's own algorithms** by walking the :mod:`cfspopcon.formulas`
  package (so adding ``formulas/foo/bar.py`` is enough — no hand-maintained import list, no
  "forgot to import it -> silently missing" failure mode), and
* **discovery of downstream-provided algorithms via entry points** (group
  ``cfspopcon.algorithms``), so an installed distribution can contribute algorithms without any
  cfspopcon-side import and without import-order coupling.

Both run lazily and exactly once, the first time the registry is queried (see
:meth:`cfspopcon.algorithm_class.Algorithm.algorithms`). The ``@Algorithm.register_algorithm``
decorator is unchanged; discovery only automates *which modules get imported*.
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
    """Run built-in + entry-point discovery exactly once (idempotent)."""
    global _discovered  # noqa: PLW0603
    if _discovered:
        return
    _discovered = True  # set first so a re-entrant lookup during discovery is a no-op
    discover_builtin_algorithms()
    load_entry_point_algorithms()
