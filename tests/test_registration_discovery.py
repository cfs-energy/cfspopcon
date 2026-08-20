"""Algorithms are discovered automatically, with no hand-maintained import list.

Replaces the previous ``test_for_anonymous_algorithms`` check, which guarded against an algorithm
being registered but not importable. The second half covers a package built on cfspopcon, which
extends the registry by walking itself.
"""

import os
import sys
from pathlib import Path

import pytest
import xarray as xr
from utils.throwaway_packages import forget_packages, write_package

from cfspopcon import algorithm_class
from cfspopcon.algorithm_class import (
    Algorithm,
    CompositeAlgorithm,
    _pending_composites,
    build_pending_composites,
    discover_algorithms_in_package,
    discover_builtin_algorithms,
)
from cfspopcon.unit_handling import Quantity, ureg


def register_probe(name):
    """Register a throwaway single-input algorithm under `name`."""
    return Algorithm.from_single_function(lambda _probe_in: _probe_in, return_keys=["_probe_out"], name=name, skip_unit_conversion=True)


def test_the_registry_is_empty_until_discovery_runs(run_script):
    """Importing cfspopcon must register nothing, so a lookup says discovery has not run.

    Run in a subprocess, since the suite discovers at session start.
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
    discover_builtin_algorithms()
    populated = dict(Algorithm.instances)
    assert len(populated) > 100
    discover_builtin_algorithms()
    assert Algorithm.instances == populated


def test_formulas_submodule_resolves_without_discovery(run_script):
    """cfspopcon.formulas.geometry must work on a bare import, with no discovery having run.

    Run in a subprocess: once discovery has bound the submodules, __getattr__ is never consulted.
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
    """A package which walks another may declare a composite spanning both walks."""
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
        discover_algorithms_in_package("_probe_outer_pkg")
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
        discover_builtin_algorithms()
        assert isinstance(Algorithm.get_algorithm(f"calc_{stem}"), Algorithm)
    finally:
        Algorithm.instances.pop(f"calc_{stem}", None)
        probe.unlink()
        # Importing it leaves bytecode behind, which would otherwise litter the package directory.
        for cached in probe.parent.glob(f"__pycache__/{stem}.*.pyc"):
            cached.unlink()
        sys.modules.pop(f"cfspopcon.formulas.{stem}", None)


def test_discover_algorithms_in_a_specified_package(tmp_path, monkeypatch, clean_composites):
    """discover_algorithms_in_package walks an arbitrary package tree and registers its algorithms."""
    write_package(
        tmp_path,
        "_walk_probe_pkg",
        {
            # Nested, and never imported by hand: only the walk can reach it.
            "models/detachment": "from cfspopcon.algorithm_class import Algorithm\n\n\n"
            "@Algorithm.register_algorithm(return_keys=['_walk_out'], skip_unit_conversion=True)\n"
            "def calc_walk_probe(_walk_in):\n"
            '    """Throwaway algorithm in a nested submodule."""\n'
            "    return _walk_in\n"
        },
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        discover_algorithms_in_package("_walk_probe_pkg")
        assert isinstance(Algorithm.get_algorithm("calc_walk_probe"), Algorithm)
    finally:
        forget_packages("_walk_probe_pkg")


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
    """A composite which cannot be built yet is reported, and stays pending until it can be."""
    CompositeAlgorithm.register_from_list(keys=["_probe_missing_base"], name="_probe_doomed")

    with pytest.raises(RuntimeError, match=r"_probe_doomed.*_probe_missing_base"):
        build_pending_composites()

    # Still pending, so a retry fails the same way rather than quietly returning a registry with the
    # composite missing -- and succeeds once the component turns up.
    assert [name for name, _ in _pending_composites] == ["_probe_doomed"]
    with pytest.raises(RuntimeError, match=r"_probe_doomed"):
        build_pending_composites()

    register_probe("_probe_missing_base")
    build_pending_composites()
    assert isinstance(Algorithm.get_algorithm("_probe_doomed"), CompositeAlgorithm)
    assert not _pending_composites


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
    """The failure names the module that broke, and does not poison discovery process-wide."""
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
            discover_algorithms_in_package("_probe_broken_pkg")

        # The guard must be cleared on the way out, or every later walk looks nested and so never
        # builds the composites it declares.
        assert not algorithm_class._walk_depth

        # An unrelated walk is unaffected: a failed walk does not poison discovery process-wide.
        discover_algorithms_in_package("_probe_innocent_pkg")
        assert isinstance(Algorithm.get_algorithm("_probe_half"), Algorithm)
    finally:
        forget_packages("_probe_broken_pkg", "_probe_innocent_pkg")


def test_a_misspelled_algorithm_name_suggests_the_real_one():
    """A near-miss lookup suggests the registered name."""
    with pytest.raises(KeyError, match="Did you mean 'calc_plasma_volume'"):
        Algorithm.get_algorithm("calc_plasma_volme")


def test_looking_up_a_declared_but_unbuilt_composite_says_so(clean_composites):
    """A composite looked up too early is a different problem from one that does not exist."""
    CompositeAlgorithm.register_from_list(keys=["calc_plasma_volume"], name="_probe_not_yet_built")
    with pytest.raises(KeyError, match="declared but not built yet"):
        Algorithm.get_algorithm("_probe_not_yet_built")


def test_the_popcon_command_discovers_before_reading_the_case(run_script):
    """popcon must discover for itself: read_case resolves the input file's algorithm names.

    Stops at read_case, since only the ordering is under test, and runs in a subprocess because the
    suite has already discovered in this one.
    """
    script = (
        "import cfspopcon.cli as cli\n"
        "from cfspopcon import Algorithm\n"
        "def stop_at_read_case(*args, **kwargs):\n"
        "    raise SystemExit(0 if len(Algorithm.instances) > 100 else 'registry not populated before read_case')\n"
        "cli.read_case = stop_at_read_case\n"
        "cli.run_popcon('example_cases/SPARC_PRD', False, {})\n"
    )
    run_script(script)


def test_the_cli_discovers_before_resolving_algorithm_names(tmp_path, run_script):
    """popcon_algorithms must discover for itself, rather than writing a near-empty file.

    Run in a subprocess: in-process this would pass whether the command discovers or not.
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


# --- A package built on cfspopcon, extending the builtin registry ---------------------------------

DOWNSTREAM = "_ds_probe_pkg"

#: A builtin composite, named rather than picked out of the registry, so which one is under test does
#: not depend on the order the walk happened to register them in.
BUILTIN_COMPOSITE = "calc_peaking_and_analytic_profiles"

#: Composites from a single algorithm through to one nested three deep, declared in a module the walk
#: reaches *before* the one defining the algorithm: none of their components exist at declaration.
DOWNSTREAM_DECLARATIONS = """\
from cfspopcon.algorithm_class import Algorithm, CompositeAlgorithm

# The point of declaring here: nothing this file names has been registered yet.
assert "calc_ds_metric" not in Algorithm.instances

CompositeAlgorithm.register_from_list(["calc_ds_metric"], name="_ds_own")
CompositeAlgorithm.register_from_list(["calc_plasma_volume", "calc_ds_metric"], name="_ds_mixed")
CompositeAlgorithm.register_from_list(["{BUILTIN_COMPOSITE}", "_ds_own"], name="_ds_of_builtin_composite")
CompositeAlgorithm.register_from_list(["_ds_of_builtin_composite", "_ds_mixed"], name="_ds_deep")
"""

#: A new variable of the package's own, so this also exercises where a downstream package declares
#: default units: extend_default_units_map, since read_default_units_from_file takes no path.
DOWNSTREAM_ALGORITHMS = """\
from cfspopcon.algorithm_class import Algorithm
from cfspopcon.unit_handling import extend_default_units_map

extend_default_units_map({"_ds_metric": "m**3"})


@Algorithm.register_algorithm(return_keys=["_ds_metric"])
def calc_ds_metric(plasma_volume):
    \"\"\"Throwaway algorithm consuming a builtin algorithm's output.\"\"\"
    return 2.0 * plasma_volume
"""

DECLARED_COMPOSITES = ["_ds_own", "_ds_mixed", "_ds_of_builtin_composite", "_ds_deep"]


@pytest.fixture()
def downstream_package(tmp_path, monkeypatch, clean_composites):
    """Walk an importable package which extends the builtin registry, and clean up after it."""
    assert isinstance(Algorithm.get_algorithm(BUILTIN_COMPOSITE), CompositeAlgorithm)
    write_package(
        tmp_path,
        DOWNSTREAM,
        {
            "a_declarations": DOWNSTREAM_DECLARATIONS.format(BUILTIN_COMPOSITE=BUILTIN_COMPOSITE),
            "z_algorithms": DOWNSTREAM_ALGORITHMS,
        },
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        discover_algorithms_in_package(DOWNSTREAM)
        yield tmp_path
    finally:
        forget_packages(DOWNSTREAM)


def test_a_walk_extends_the_builtin_registry(downstream_package):
    """Every declared composite builds, including ones nested three deep across both packages."""
    for name in DECLARED_COMPOSITES:
        assert isinstance(Algorithm.get_algorithm(name), CompositeAlgorithm), name

    # Flattening reaches through both packages: the deepest composite runs the builtin composite's
    # algorithms as well as the new one.
    member_names = [alg.name for alg in Algorithm.get_algorithm("_ds_deep").algorithms]
    assert "calc_ds_metric" in member_names
    builtin_members = [alg.name for alg in Algorithm.get_algorithm(BUILTIN_COMPOSITE).algorithms]
    assert set(builtin_members) <= set(member_names)


def test_a_downstream_composite_runs_end_to_end(downstream_package):
    """A composite mixing builtin and new algorithms computes the new variable, in its own units."""
    inputs = xr.Dataset(
        {
            "major_radius": Quantity(1.85, ureg.m),
            "inverse_aspect_ratio": Quantity(0.3, ureg.dimensionless),
            "areal_elongation": Quantity(1.75, ureg.dimensionless),
        }
    )
    result = Algorithm.get_algorithm("_ds_mixed").update_dataset(inputs)

    assert result["_ds_metric"] == 2.0 * result["plasma_volume"]
    assert result["_ds_metric"].pint.units == ureg.m**3


def test_the_machinery_works_without_the_builtin_algorithms(run_script):
    """A package may use Algorithm/CompositeAlgorithm without ever discovering cfspopcon's own.

    Run in a subprocess, to assert that the registry holds nothing but its own algorithms.
    """
    script = """
import xarray as xr
from cfspopcon.algorithm_class import Algorithm, CompositeAlgorithm, build_pending_composites
from cfspopcon.unit_handling import Quantity, extend_default_units_map, ureg

extend_default_units_map({"_solo_area": "m**2"})


@Algorithm.register_algorithm(return_keys=["_solo_area"])
def calc_solo_area(_solo_width, _solo_height):
    return _solo_width * _solo_height


@Algorithm.register_algorithm(return_keys=["_solo_label"], skip_unit_conversion=True)
def calc_solo_label(_solo_area):
    return "big" if _solo_area > Quantity(1.0, ureg.m**2) else "small"


CompositeAlgorithm.register_from_list(["calc_solo_area", "calc_solo_label"], name="_solo_composite")
build_pending_composites()

assert set(Algorithm.instances) == {"calc_solo_area", "calc_solo_label", "_solo_composite"}, Algorithm.instances

composite = Algorithm.get_algorithm("_solo_composite")
inputs = xr.Dataset({"_solo_width": Quantity(2.0, ureg.m), "_solo_height": Quantity(3.0, ureg.m)})
assert composite.validate_inputs(inputs, quiet=True)
result = composite.update_dataset(inputs)
assert result["_solo_area"] == Quantity(6.0, ureg.m**2)
assert result["_solo_label"] == "big"
"""
    run_script(script)
