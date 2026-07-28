"""Algorithms are discovered automatically, with no hand-maintained import list.

Replaces the previous ``test_for_anonymous_algorithms`` check: once discovery walks the package,
the "registered-but-not-importable" gap it guarded against can no longer occur.
"""

import os
import sys
from pathlib import Path

import pytest
from utils.throwaway_packages import forget_packages, write_package

from cfspopcon import _discovery
from cfspopcon.algorithm_class import Algorithm, CompositeAlgorithm, _pending_composites, build_pending_composites


def register_probe(name):
    """Register a throwaway single-input algorithm under `name`."""
    return Algorithm.from_single_function(lambda _probe_in: _probe_in, return_keys=["_probe_out"], name=name, skip_unit_conversion=True)


def test_the_registry_is_empty_until_discovery_runs(run_script):
    """Importing cfspopcon must register nothing, so a lookup says discovery has not run.

    Run in a subprocess: the suite discovers once at session start, so this is unobservable here.
    An empty registry reads as "nothing discovered yet"; a registry holding whatever happened to be
    imported would look like a real one that is missing algorithms.
    """
    script = (
        "import cfspopcon\n"
        "from cfspopcon import Algorithm\n"
        "assert Algorithm.instances == {}, Algorithm.instances\n"
        "assert Algorithm.algorithms() == []\n"
        "assert not hasattr(cfspopcon, 'AtomicData'), 'importing AtomicData would register read_atomic_data'\n"
        "try:\n"
        "    cfspopcon.registry['calc_plasma_volume']\n"
        "except KeyError as exc:\n"
        "    assert 'discovery has not run' in str(exc), exc\n"
        "else:\n"
        "    raise AssertionError('lookup should have failed')\n"
        "cfspopcon.discover_builtin_algorithms()\n"
        "assert len(Algorithm.instances) > 100\n"
    )
    run_script(script)


def test_repeated_discovery_changes_nothing():
    """Calling discovery again must not trip the already-registered guard."""
    _discovery.discover_builtin_algorithms()
    populated = dict(Algorithm.instances)
    assert len(populated) > 100
    _discovery.discover_builtin_algorithms()
    assert Algorithm.instances == populated


def test_formulas_submodule_resolves_without_discovery(run_script):
    """cfspopcon.formulas.geometry must work on a bare import, with no discovery having run.

    Run in a subprocess: discovery imports every submodule and binds it as a real attribute, so
    __getattr__ would never be consulted once the suite's session fixture has run.
    """
    script = (
        "import cfspopcon\n"
        "from cfspopcon import Algorithm, formulas\n"
        "assert 'geometry' in dir(formulas) and '__name__' in dir(formulas)\n"
        "assert formulas.geometry.analytical.calc_plasma_volume is not None\n"
        # Importing that submodule runs its own decorators, but must not pull in the whole package.
        "assert len(Algorithm.instances) < 100, len(Algorithm.instances)\n"
    )
    run_script(script)


def test_a_nested_walk_leaves_the_composite_build_to_the_outermost_one(tmp_path, monkeypatch, clean_composites):
    """A provider which walks its own package may declare a composite spanning both walks.

    The inner walk must not build, or it would fail on a component the outer walk has not reached.
    """
    write_package(
        tmp_path,
        "_probe_inner_pkg",
        {
            "m": "from cfspopcon.algorithm_class import Algorithm, CompositeAlgorithm\n"
            "Algorithm.from_single_function(lambda x: x, return_keys=['y'], name='_probe_inner', skip_unit_conversion=True)\n"
            "CompositeAlgorithm.register_from_list(['_probe_inner', '_probe_outer'], name='_probe_spanning')\n"
        },
    )
    write_package(
        tmp_path,
        "_probe_outer_pkg",
        {
            "m": "from cfspopcon.algorithm_class import Algorithm\n"
            "from cfspopcon import discover_algorithms_in_package\n"
            "discover_algorithms_in_package('_probe_inner_pkg')\n"
            "Algorithm.from_single_function(lambda x: x, return_keys=['y'], name='_probe_outer', skip_unit_conversion=True)\n"
        },
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        _discovery.discover_algorithms_in_package("_probe_outer_pkg")
        assert isinstance(Algorithm.get_algorithm("_probe_spanning"), CompositeAlgorithm)
    finally:
        forget_packages("_probe_inner_pkg", "_probe_outer_pkg")


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
    register_probe("_probe_base")

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

    register_probe("_probe_late_base")
    build_pending_composites()
    assert isinstance(Algorithm.get_algorithm("_probe_late_composite"), CompositeAlgorithm)
    assert not _pending_composites


def test_a_broken_entry_point_fails_discovery_loudly(fake_entry_points):
    """A provider whose target raises must not be papered over: discovery propagates it."""

    def broken():
        raise ImportError("this distribution is broken")

    fake_entry_points(broken)
    with pytest.raises(ImportError, match="this distribution is broken"):
        _discovery.discover_builtin_algorithms()


def test_a_composite_that_fails_to_build_does_not_take_the_others_with_it(clean_composites):
    """A declaration is dropped from the pending list only once it has actually been built."""
    register_probe("_probe_component")
    # The first collides with an already-registered name; the second is perfectly buildable.
    CompositeAlgorithm.register_from_list(keys=["_probe_component"], name="calc_plasma_volume")
    CompositeAlgorithm.register_from_list(keys=["_probe_component"], name="_probe_survivor")

    with pytest.raises(RuntimeError, match="already registered"):
        build_pending_composites()

    assert "_probe_survivor" in [name for name, _ in _pending_composites]
    with pytest.raises(RuntimeError, match="already registered"):
        build_pending_composites()  # still fails the same way rather than quietly succeeding


def test_a_walk_which_raises_blames_the_broken_package_only(tmp_path, monkeypatch, clean_composites):
    """The failure names the module that broke, and does not poison discovery process-wide.

    Retrying the same walk in the same process is not supported: whatever the abandoned attempt
    registered is still registered, so the retry trips the already-registered guard instead of
    reporting the original cause. Fix the cause and start a new process.
    """
    write_package(
        tmp_path,
        "_probe_broken_pkg",
        {
            "__init__": (
                "from cfspopcon.algorithm_class import Algorithm\n"
                "Algorithm.from_single_function(lambda x: x, return_keys=['y'], name='_probe_half', skip_unit_conversion=True)\n"
                "raise ImportError('an optional dependency is missing')\n"
            )
        },
    )
    write_package(tmp_path, "_probe_innocent_pkg", {"m": "x = 1\n"})
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        with pytest.raises(ImportError, match="optional dependency"):
            _discovery.discover_algorithms_in_package("_probe_broken_pkg")

        # The guard must be cleared on the way out, or every later walk looks nested and so never
        # builds the composites it declares.
        assert not _discovery._walking

        # An unrelated walk is unaffected: a failed walk does not poison discovery process-wide.
        _discovery.discover_algorithms_in_package("_probe_innocent_pkg")
        assert isinstance(Algorithm.get_algorithm("_probe_half"), Algorithm)
    finally:
        forget_packages("_probe_broken_pkg", "_probe_innocent_pkg")


def test_a_package_name_which_does_not_resolve_leaves_the_registry_usable():
    """A typo has run no code, so it must not disturb what is already registered."""
    with pytest.raises(ModuleNotFoundError):
        _discovery.discover_algorithms_in_package("_probe_no_such_package")
    assert len(Algorithm.algorithms()) > 100


def test_a_composite_missing_a_component_is_completed_by_a_later_walk(tmp_path, monkeypatch, clean_composites):
    """An unbuildable composite is recoverable, unlike a walk that raised: the declaration waits."""
    write_package(
        tmp_path,
        "_probe_declarer",
        {
            "m": "from cfspopcon.algorithm_class import CompositeAlgorithm\nCompositeAlgorithm.register_from_list(['_probe_supplied'], name='_probe_waiting')\n"
        },
    )
    write_package(
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
        forget_packages("_probe_declarer", "_probe_supplier")


def test_a_misspelled_algorithm_name_suggests_the_real_one():
    """The commonest mistake deserves a pointer, not a lecture about entry points."""
    with pytest.raises(KeyError, match="Did you mean 'calc_plasma_volume'"):
        Algorithm.get_algorithm("calc_plasma_volme")


def test_looking_up_a_declared_but_unbuilt_composite_says_so(clean_composites):
    """A composite looked up too early is a different problem from one that does not exist."""
    CompositeAlgorithm.register_from_list(keys=["calc_plasma_volume"], name="_probe_not_yet_built")
    with pytest.raises(KeyError, match="declared but not built yet"):
        Algorithm.get_algorithm("_probe_not_yet_built")


def test_the_cli_discovers_before_resolving_algorithm_names(tmp_path, run_script):
    """popcon_algorithms must discover for itself, rather than writing a near-empty file.

    Run in a subprocess: in-process the suite has already discovered, so this would pass whether
    the command populates the registry or not.
    """
    output = tmp_path / "algorithms.yaml"
    script = (
        "from click.testing import CliRunner\n"
        "from cfspopcon.cli import write_algorithms_yaml\n"
        f"result = CliRunner().invoke(write_algorithms_yaml, ['--output', {str(output)!r}])\n"
        # CliRunner captures whatever the command raised rather than letting it out.
        "assert result.exit_code == 0, result.exception or result.output\n"
    )
    run_script(script)
    entries = [line for line in output.read_text().splitlines() if line and not line.startswith((" ", "#"))]
    assert len(entries) > 100, entries
