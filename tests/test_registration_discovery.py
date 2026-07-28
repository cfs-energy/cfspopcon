"""Algorithms are discovered automatically, with no hand-maintained import list.

Replaces the previous ``test_for_anonymous_algorithms`` check: once discovery walks the package,
the "registered-but-not-importable" gap it guarded against can no longer occur.
"""

import os
import subprocess
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import pytest

import cfspopcon
from cfspopcon import _discovery
from cfspopcon.algorithm_class import Algorithm, CompositeAlgorithm, _pending_composites, build_pending_composites


@pytest.fixture()
def clean_composites():
    """Isolate a test's registrations and declarations from the rest of the session.

    cfspopcon's own modules declare composites at import time, and pytest imports every test module
    during collection, so the pending list is generally non-empty before any test runs. Set it
    aside for the duration rather than building or discarding those declarations, and undo whatever
    the test registers so it cannot depend on, or leak into, another test.
    """
    Algorithm.algorithms()  # settle discovery first, so the snapshot below is of a full registry
    saved = _pending_composites[:]
    registered = set(Algorithm.instances)
    _pending_composites.clear()
    yield
    # Diff the registry rather than asking each test to declare what it added: a test which fails
    # partway still registered whatever it got to, and that must not leak into the next one.
    for name in set(Algorithm.instances) - registered:
        del Algorithm.instances[name]
    _pending_composites[:] = saved
    if saved and _discovery._discovered:
        # Discovery completed while the real declarations were set aside, so nothing else will
        # build them now. Do it here, or they stay pending for the rest of the session.
        build_pending_composites()


@pytest.fixture()
def fake_entry_points(monkeypatch):
    """Install fake entry points, each loading to one of the given targets."""

    def install(*targets):
        eps = [SimpleNamespace(load=lambda target=target: target) for target in targets]
        monkeypatch.setattr(_discovery, "entry_points", lambda group: eps)

    return install


@pytest.fixture()
def fresh_discovery(monkeypatch):
    """Rewind discovery so a test can drive it from the start, and restore it afterwards."""
    for flag, value in (("_discovered", False), ("_discovering", False), ("_failure", None)):
        monkeypatch.setattr(_discovery, flag, value)


def test_discovery_is_idempotent_and_populates_registry(monkeypatch):
    # The first call populates the registry; a second call must be a no-op.
    _discovery.ensure_discovered()
    populated = dict(Algorithm.instances)
    assert len(populated) > 100
    _discovery.ensure_discovered()
    assert Algorithm.instances == populated

    # Comparing contents is not enough: sys.modules would hide a repeated walk. The latch has to
    # stop the work happening at all, or every registry query re-walks the package.
    walks = []
    monkeypatch.setattr(_discovery, "discover_builtin_algorithms", lambda: walks.append(True))
    _discovery.ensure_discovered()
    Algorithm.algorithms()
    assert walks == []


def test_formulas_submodule_resolves_before_the_registry_is_queried():
    """cfspopcon.formulas.geometry must work on a bare import, with no discovery having run.

    Run in a subprocess: any earlier test in this session triggers discovery, which imports every
    submodule and binds it as a real attribute, so __getattr__ would never be consulted here.
    """
    script = (
        "import cfspopcon\n"
        "from cfspopcon import _discovery, formulas\n"
        "assert not _discovery._discovered, 'discovery should not have run on a bare import'\n"
        "assert 'geometry' in dir(formulas) and '__name__' in dir(formulas)\n"
        "assert formulas.geometry.analytical.calc_plasma_volume is not None\n"
        "assert not _discovery._discovered, 'attribute access must not trigger discovery'\n"
    )
    subprocess.run([sys.executable, "-c", script], check=True, cwd=Path(cfspopcon.__file__).parents[1])


def test_a_failed_discovery_keeps_failing_the_same_way(monkeypatch, fresh_discovery):
    """Discovery is all-or-nothing: the failure is remembered, not quietly half-applied."""

    def boom():
        raise ImportError("a formulas module failed to import")

    monkeypatch.setattr(_discovery, "discover_builtin_algorithms", boom)
    with pytest.raises(ImportError, match="failed to import") as first:
        _discovery.ensure_discovered()

    # Even once the cause is gone, the original error surfaces again rather than a registry which
    # looks complete but is missing whatever the failed attempt never got to.
    monkeypatch.setattr(_discovery, "discover_builtin_algorithms", lambda: None)
    with pytest.raises(ImportError) as second:
        Algorithm.algorithms()
    assert second.value is first.value

    # Re-raising the stored object would append a frame to its traceback every time, so a
    # long-lived process querying a poisoned registry would grow one without bound.
    def depth():
        with pytest.raises(ImportError) as raised:
            Algorithm.algorithms()
        return len(traceback.format_exception(raised.value))

    assert depth() == depth()


def test_reentrant_discovery_does_not_restart_the_walk(monkeypatch, fresh_discovery):
    """A registry query from a module being imported by the walk must not recurse."""
    calls = []

    def walk():
        calls.append(True)
        _discovery.ensure_discovered()  # re-entrant, as an import-time get_algorithm would be

    monkeypatch.setattr(_discovery, "discover_builtin_algorithms", walk)
    _discovery.ensure_discovered()
    assert calls == [True]


def test_drop_in_module_is_discovered_without_editing_init():
    """A brand-new formulas submodule is found by the pkgutil walk with no __init__.py edit."""
    from cfspopcon import formulas

    # Unique per process, so a concurrent or crashed run cannot collide with this one.
    stem = f"_probe_drop_in_{os.getpid()}"
    probe = Path(formulas.__file__).parent / f"{stem}.py"
    probe.write_text(
        "from cfspopcon.algorithm_class import Algorithm\n\n\n"
        f"@Algorithm.register_algorithm(return_keys=['{stem}_out'], skip_unit_conversion=True)\n"
        f"def calc_{stem}(_probe_in):\n"
        '    """Throwaway probe algorithm."""\n'
        "    return _probe_in\n"
    )
    try:
        _discovery.discover_builtin_algorithms()
        assert isinstance(Algorithm.get_algorithm(f"calc_{stem}"), Algorithm)
    finally:
        Algorithm.instances.pop(f"calc_{stem}", None)
        probe.unlink()
        # Importing it leaves bytecode behind, which would otherwise litter the package directory.
        for cached in probe.parent.glob(f"__pycache__/{stem}.*.pyc"):
            cached.unlink()
        sys.modules.pop(f"cfspopcon.formulas.{stem}", None)


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


def test_entry_point_callable_is_invoked(fake_entry_points):
    """A downstream entry point whose target is a callable is invoked to register (no cfspopcon import)."""
    called = []
    fake_entry_points(lambda: called.append(True))
    _discovery.load_entry_point_algorithms()
    assert called == [True]


def test_formulas_rejects_an_unknown_attribute():
    """__getattr__ must raise AttributeError, not a ModuleNotFoundError from the failed import."""
    from cfspopcon import formulas

    with pytest.raises(AttributeError):
        formulas.not_a_subpackage


def test_composites_build_regardless_of_declaration_order(clean_composites):
    """A composite may be declared before the algorithms and composites it is built from."""

    # Declared first, but depends on a composite which is itself declared after its own component.
    CompositeAlgorithm.register_from_list(keys=["_probe_composite"], name="_probe_of_composite")
    CompositeAlgorithm.register_from_list(keys=["_probe_base"], name="_probe_composite")
    Algorithm.from_single_function(lambda _probe_in: _probe_in, return_keys=["_probe_out"], name="_probe_base", skip_unit_conversion=True)

    build_pending_composites()

    assert not _pending_composites
    assert isinstance(Algorithm.get_algorithm("_probe_composite"), CompositeAlgorithm)
    assert isinstance(Algorithm.get_algorithm("_probe_of_composite"), CompositeAlgorithm)


def test_unsatisfiable_composite_names_its_missing_components(clean_composites):
    """A composite which can never be built is reported, rather than silently left unbuilt."""
    CompositeAlgorithm.register_from_list(keys=["_probe_never_registered"], name="_probe_doomed")

    with pytest.raises(RuntimeError, match=r"_probe_doomed.*_probe_never_registered"):
        build_pending_composites()

    # The declaration stays pending, so a retry fails the same way instead of quietly returning a
    # registry with the composite missing.
    assert [name for name, _ in _pending_composites] == ["_probe_doomed"]
    with pytest.raises(RuntimeError, match=r"_probe_doomed"):
        build_pending_composites()


def test_pending_composite_builds_once_a_later_step_registers_its_component(clean_composites):
    """A declaration left pending by a failed build is satisfied when the component turns up."""
    CompositeAlgorithm.register_from_list(keys=["_probe_late_base"], name="_probe_late_composite")

    with pytest.raises(RuntimeError, match=r"_probe_late_composite"):
        build_pending_composites()

    Algorithm.from_single_function(
        lambda _probe_in: _probe_in, return_keys=["_probe_out"], name="_probe_late_base", skip_unit_conversion=True
    )
    build_pending_composites()
    assert isinstance(Algorithm.get_algorithm("_probe_late_composite"), CompositeAlgorithm)
    assert not _pending_composites


def test_a_broken_entry_point_fails_the_query_loudly(fake_entry_points, fresh_discovery):
    """An unloadable provider must not be papered over: the registry query says so."""

    def broken():
        raise ImportError("this distribution is broken")

    fake_entry_points(broken)
    with pytest.raises(ImportError, match="this distribution is broken"):
        _discovery.ensure_discovered()


def test_a_composite_that_fails_to_build_does_not_take_the_others_with_it(clean_composites):
    """A declaration is dropped from the pending list only once it has actually been built."""
    Algorithm.from_single_function(
        lambda _probe_in: _probe_in, return_keys=["_probe_out"], name="_probe_component", skip_unit_conversion=True
    )
    # The first collides with an already-registered name; the second is perfectly buildable.
    CompositeAlgorithm.register_from_list(keys=["_probe_component"], name="calc_plasma_volume")
    CompositeAlgorithm.register_from_list(keys=["_probe_component"], name="_probe_survivor")

    with pytest.raises(RuntimeError, match="already registered"):
        build_pending_composites()

    assert "_probe_survivor" in [name for name, _ in _pending_composites]
    with pytest.raises(RuntimeError, match="already registered"):
        build_pending_composites()  # still fails the same way rather than quietly succeeding


def _write_package(root, name, modules):
    """Create an importable package under `root`, as {submodule name: source}."""
    pkg = root / name
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text(modules.pop("__init__", ""))
    for module, source in modules.items():
        (pkg / f"{module}.py").write_text(source)
    return pkg


def _forget(*package_names):
    for name in [m for m in sys.modules if m.split(".")[0] in package_names]:
        sys.modules.pop(name, None)


def test_a_package_which_fails_to_import_is_blamed_rather_than_the_next_caller(tmp_path, monkeypatch, fresh_discovery):
    """A half-walked package poisons the registry, and later callers are told why, not blamed."""
    _write_package(
        tmp_path,
        "_probe_broken_pkg",
        {
            "__init__": (
                "from cfspopcon.algorithm_class import Algorithm, CompositeAlgorithm\n"
                "Algorithm.from_single_function(lambda x: x, return_keys=['y'], name='_probe_half', skip_unit_conversion=True)\n"
                "CompositeAlgorithm.register_from_list(['_probe_half'], name='_probe_orphan')\n"
                "raise ImportError('an optional dependency is missing')\n"
            )
        },
    )
    _write_package(tmp_path, "_probe_innocent_pkg", {"m": "x = 1\n"})
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        with pytest.raises(ImportError, match="optional dependency"):
            _discovery.discover_algorithms_in_package("_probe_broken_pkg")

        # The innocent package's own walk is fine, so it must not be blamed for the orphan.
        with pytest.raises(ImportError, match="optional dependency"):
            _discovery.discover_algorithms_in_package("_probe_innocent_pkg")
    finally:
        for name in ("_probe_half", "_probe_orphan"):
            Algorithm.instances.pop(name, None)
        _pending_composites[:] = [entry for entry in _pending_composites if entry[0] != "_probe_orphan"]
        _forget("_probe_broken_pkg", "_probe_innocent_pkg")


def test_a_package_name_which_does_not_resolve_leaves_the_registry_usable(fresh_discovery):
    """A typo has run no code, so it must not poison discovery for the rest of the process."""
    with pytest.raises(ModuleNotFoundError):
        _discovery.discover_algorithms_in_package("_probe_no_such_package")
    assert len(Algorithm.algorithms()) > 100


def test_a_composite_missing_a_component_is_completed_by_a_later_walk(tmp_path, monkeypatch, clean_composites):
    """An unbuildable composite is recoverable, unlike a walk that raised: the declaration waits."""
    _write_package(
        tmp_path,
        "_probe_declarer",
        {
            "m": "from cfspopcon.algorithm_class import CompositeAlgorithm\nCompositeAlgorithm.register_from_list(['_probe_supplied'], name='_probe_waiting')\n"
        },
    )
    _write_package(
        tmp_path,
        "_probe_supplier",
        {
            "m": "from cfspopcon.algorithm_class import Algorithm\n"
            "Algorithm.from_single_function(lambda x: x, return_keys=['y'], name='_probe_supplied', skip_unit_conversion=True)\n"
        },
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        with pytest.raises(RuntimeError, match="_probe_waiting"):
            _discovery.discover_algorithms_in_package("_probe_declarer")
        _discovery.discover_algorithms_in_package("_probe_supplier")
        assert isinstance(Algorithm.get_algorithm("_probe_waiting"), CompositeAlgorithm)
    finally:
        _forget("_probe_declarer", "_probe_supplier")


def test_a_misspelled_algorithm_name_suggests_the_real_one():
    """The commonest mistake deserves a pointer, not a lecture about entry points."""
    with pytest.raises(KeyError, match="Did you mean 'calc_plasma_volume'"):
        Algorithm.get_algorithm("calc_plasma_volme")


def test_looking_up_a_declared_but_unbuilt_composite_says_so(clean_composites):
    """A composite looked up too early is a different problem from one that does not exist."""
    CompositeAlgorithm.register_from_list(keys=["calc_plasma_volume"], name="_probe_not_yet_built")
    with pytest.raises(KeyError, match="declared but not built yet"):
        Algorithm.get_algorithm("_probe_not_yet_built")
