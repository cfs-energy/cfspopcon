"""Algorithms are discovered automatically, with no hand-maintained import list.

Replaces the previous ``test_for_anonymous_algorithms`` check: once discovery walks the package,
the "registered-but-not-importable" gap it guarded against can no longer occur.
"""

import sys
from pathlib import Path

from cfspopcon import _discovery
from cfspopcon.algorithm_class import Algorithm


def test_discovery_is_idempotent_and_populates_registry():
    # The first call populates the registry; a second call must be a no-op.
    _discovery.ensure_discovered()
    populated = dict(Algorithm.instances)
    assert len(populated) > 100
    _discovery.ensure_discovered()
    assert Algorithm.instances == populated


def test_drop_in_module_is_discovered_without_editing_init():
    """A brand-new formulas submodule is found by the pkgutil walk with no __init__.py edit."""
    from cfspopcon import formulas

    probe = Path(formulas.__file__).parent / "_probe_drop_in.py"
    probe.write_text(
        "from cfspopcon.algorithm_class import Algorithm\n\n\n"
        "@Algorithm.register_algorithm(return_keys=['_probe_out'], skip_unit_conversion=True)\n"
        "def calc_probe(_probe_in):\n"
        '    """Throwaway probe algorithm."""\n'
        "    return _probe_in\n"
    )
    try:
        _discovery.discover_builtin_algorithms()
        assert isinstance(Algorithm.get_algorithm("calc_probe"), Algorithm)
    finally:
        Algorithm.instances.pop("calc_probe", None)
        probe.unlink()
        sys.modules.pop("cfspopcon.formulas._probe_drop_in", None)


def test_discover_algorithms_in_a_specified_package(tmp_path, monkeypatch):
    """discover_algorithms_in_package walks an arbitrary package tree and registers its algorithms."""
    pkg = tmp_path / "_walk_probe_pkg"
    submodule_dir = pkg / "models"
    submodule_dir.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (submodule_dir / "__init__.py").write_text("")
    (submodule_dir / "detachment.py").write_text(
        "from cfspopcon.algorithm_class import Algorithm\n\n\n"
        "@Algorithm.register_algorithm(return_keys=['_walk_out'], skip_unit_conversion=True)\n"
        "def calc_walk_probe(_walk_in):\n"
        '    """Throwaway algorithm in a nested submodule."""\n'
        "    return _walk_in\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        # The algorithm lives in a nested submodule that is never imported by hand.
        _discovery.discover_algorithms_in_package("_walk_probe_pkg")
        assert isinstance(Algorithm.get_algorithm("calc_walk_probe"), Algorithm)
    finally:
        Algorithm.instances.pop("calc_walk_probe", None)
        for name in [m for m in sys.modules if m == "_walk_probe_pkg" or m.startswith("_walk_probe_pkg.")]:
            sys.modules.pop(name, None)


def test_entry_point_callable_is_invoked(monkeypatch):
    """A downstream entry point whose target is a callable is invoked to register (no cfspopcon import)."""
    called = []

    class _FakeEntryPoint:
        def load(self):
            return lambda: called.append(True)

    monkeypatch.setattr(_discovery, "entry_points", lambda group: [_FakeEntryPoint()])
    _discovery.load_entry_point_algorithms()
    assert called == [True]
