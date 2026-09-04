"""Algorithms are discovered automatically, with no hand-maintained import list.

The second half covers a package built on cfspopcon, which extends the registry by walking itself.
"""

import importlib
import os
import sys
from pathlib import Path

import pytest
import xarray as xr
from utils.throwaway_packages import forget_packages, write_package

from cfspopcon.algorithm_class import (
    Algorithm,
    CompositeAlgorithm,
    discover_builtin_algorithms,
    register_plugin,
)
from cfspopcon.unit_handling import Quantity, ureg


def register_probe(name):
    """Register a throwaway single-input algorithm under `name`."""
    algorithm = Algorithm.from_single_function(
        lambda _probe_in: _probe_in, return_keys=["_probe_out"], name=name, skip_unit_conversion=True
    )
    Algorithm.register(algorithm)
    return algorithm


def test_a_bare_import_registers_nothing_and_first_use_discovers(run_script):
    """Importing cfspopcon registers nothing; the first use of the registry registers the builtins.

    Run in a subprocess, since the suite discovers at session start.
    """
    script = (
        "import cfspopcon\n"
        "from cfspopcon import Algorithm\n"
        "assert Algorithm.instances == {}, Algorithm.instances\n"
        "cfspopcon.registry['calc_plasma_volume']\n"
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


def test_browsing_formulas_registers_nothing(run_script):
    """cfspopcon.formulas is an ordinary package; browsing it must leave the registry empty.

    Run in a subprocess, since the suite discovers at session start.
    """
    script = (
        "import cfspopcon\n"
        "from cfspopcon import Algorithm, formulas\n"
        "assert 'geometry' in dir(formulas)\n"
        "assert formulas.geometry.analytical.calc_plasma_volume is not None\n"
        "assert Algorithm.instances == {}, Algorithm.instances\n"
    )
    run_script(script)


def test_a_composite_may_span_packages_registered_earlier(tmp_path, monkeypatch, clean_composites):
    """Registration is sequential: a composite may name anything registered by the end of its own package."""
    write_package(
        tmp_path,
        "_probe_first_pkg",
        {
            "m": "from cfspopcon.algorithm_class import Algorithm\n"
            "_probe_first = Algorithm.from_single_function(lambda x: x, return_keys=['y'], name='_probe_first', skip_unit_conversion=True)\n"
        },
    )
    write_package(
        tmp_path,
        "_probe_second_pkg",
        {
            "m": "from cfspopcon.algorithm_class import Algorithm, CompositeAlgorithm\n"
            "_probe_second = Algorithm.from_single_function(lambda x: x, return_keys=['y'], name='_probe_second', skip_unit_conversion=True)\n"
            "_probe_spanning = CompositeAlgorithm.declare(['_probe_first', '_probe_second'], name='_probe_spanning')\n"
        },
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        register_plugin("_probe_first_pkg")
        register_plugin("_probe_second_pkg")
        assert isinstance(Algorithm.get_algorithm("_probe_spanning"), CompositeAlgorithm)
    finally:
        forget_packages("_probe_first_pkg", "_probe_second_pkg")


def test_a_composite_may_not_span_a_package_registered_later(tmp_path, monkeypatch, clean_composites):
    """A composite naming a not-yet-registered algorithm fails at its own package's registration."""
    write_package(
        tmp_path,
        "_probe_early_pkg",
        {
            "m": "from cfspopcon.algorithm_class import CompositeAlgorithm\n"
            "_probe_premature = CompositeAlgorithm.declare(['_probe_late'], name='_probe_premature')\n"
        },
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        with pytest.raises(RuntimeError, match=r"_probe_premature.*_probe_late"):
            register_plugin("_probe_early_pkg")
    finally:
        forget_packages("_probe_early_pkg")


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


def test_registering_a_plugin_walks_nested_submodules(tmp_path, monkeypatch, clean_composites):
    """register_plugin walks the whole package tree, registering algorithms from nested submodules."""
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
        register_plugin("_walk_probe_pkg")
        assert isinstance(Algorithm.get_algorithm("calc_walk_probe"), Algorithm)
    finally:
        forget_packages("_walk_probe_pkg")


def test_composites_build_regardless_of_declaration_order(tmp_path, monkeypatch, clean_composites):
    """A composite may be declared before the algorithms and composites it is built from."""
    write_package(
        tmp_path,
        "_probe_order2_pkg",
        {
            # Walked first: declares a composite of a composite, then the composite, before the base exists.
            "aaa_declarations": "from cfspopcon.algorithm_class import CompositeAlgorithm\n"
            "_probe_of_composite = CompositeAlgorithm.declare(['_probe_composite'], name='_probe_of_composite')\n"
            "_probe_composite = CompositeAlgorithm.declare(['_probe_base'], name='_probe_composite')\n",
            "zzz_base": "from cfspopcon.algorithm_class import Algorithm\n"
            "_probe_base = Algorithm.from_single_function(lambda x: x, return_keys=['y'], name='_probe_base', skip_unit_conversion=True)\n",
        },
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        register_plugin("_probe_order2_pkg")
        assert isinstance(Algorithm.get_algorithm("_probe_composite"), CompositeAlgorithm)
        assert isinstance(Algorithm.get_algorithm("_probe_of_composite"), CompositeAlgorithm)
    finally:
        forget_packages("_probe_order2_pkg")


def test_a_declared_composite_may_override_a_registered_name(tmp_path, monkeypatch, clean_composites):
    """declare(override=True) replaces a registered algorithm of the same name, mirroring the decorator's flag."""
    write_package(
        tmp_path,
        "_probe_ov_first_pkg",
        {
            "m": "from cfspopcon.algorithm_class import Algorithm\n"
            "_probe_ov_target = Algorithm.from_single_function(lambda x: x, return_keys=['y'], name='_probe_ov_target', skip_unit_conversion=True)\n"
        },
    )
    write_package(
        tmp_path,
        "_probe_ov_second_pkg",
        {
            "m": "from cfspopcon.algorithm_class import Algorithm, CompositeAlgorithm\n"
            "_probe_ov_part = Algorithm.from_single_function(lambda x: x, return_keys=['y'], name='_probe_ov_part', skip_unit_conversion=True)\n"
            "_probe_ov = CompositeAlgorithm.declare(['_probe_ov_part'], name='_probe_ov_target', override=True)\n"
        },
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        register_plugin("_probe_ov_first_pkg")
        assert not isinstance(Algorithm.get_algorithm("_probe_ov_target"), CompositeAlgorithm)
        register_plugin("_probe_ov_second_pkg")
        assert isinstance(Algorithm.get_algorithm("_probe_ov_target"), CompositeAlgorithm)
    finally:
        forget_packages("_probe_ov_first_pkg", "_probe_ov_second_pkg")


def test_a_walk_which_raises_blames_the_broken_package_only(tmp_path, monkeypatch, clean_composites):
    """The failure names the module that broke, is rolled back, and does not affect later registration."""
    write_package(
        tmp_path,
        "_probe_broken_pkg",
        {
            "__init__": (
                "from cfspopcon.algorithm_class import Algorithm\n"
                "_probe_half = Algorithm.from_single_function(lambda x: x, return_keys=['y'], name='_probe_half', skip_unit_conversion=True)\n"
                "raise ImportError('an optional dependency is missing')\n"
            )
        },
    )
    write_package(tmp_path, "_probe_innocent_pkg", {"m": "x = 1\n"})
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        with pytest.raises(ImportError, match="optional dependency"):
            register_plugin("_probe_broken_pkg")

        # Rolled back: the half-registered algorithm is gone, and an unrelated walk is unaffected.
        assert "_probe_half" not in Algorithm.instances
        assert "_probe_broken_pkg" not in sys.modules
        register_plugin("_probe_innocent_pkg")
    finally:
        forget_packages("_probe_broken_pkg", "_probe_innocent_pkg")


def test_a_failed_registration_rolls_everything_back(tmp_path, monkeypatch, clean_composites):
    """A failure anywhere in the package undoes its algorithms and units."""
    from cfspopcon.unit_handling import default_units

    monkeypatch.setattr(default_units, "_DEFAULT_UNIT_BY_VARIABLE", dict(default_units._DEFAULT_UNIT_BY_VARIABLE))
    write_package(
        tmp_path,
        "_probe_atomic_pkg",
        {
            "aaa_good": "from cfspopcon.algorithm_class import Algorithm\n"
            "_probe_atomic = Algorithm.from_single_function(lambda x: x, return_keys=['y'], name='_probe_atomic', skip_unit_conversion=True)\n",
            "zzz_broken": "raise ImportError('broken on purpose')\n",
        },
    )
    (tmp_path / "_probe_atomic_pkg" / "variables.yaml").write_text(
        "_probe_atomic_var:\n  default_units: m**3\n  description:\n  - A probe variable.\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        with pytest.raises(ImportError, match="broken on purpose"):
            register_plugin("_probe_atomic_pkg")

        assert "_probe_atomic" not in Algorithm.instances
        assert "_probe_atomic_var" not in default_units.default_units_map()
    finally:
        forget_packages("_probe_atomic_pkg")


def test_a_failed_registration_can_be_retried_after_fixing(tmp_path, monkeypatch, clean_composites):
    """Fixing the broken module and registering again works in the same process."""
    pkg = write_package(
        tmp_path,
        "_probe_retry_pkg",
        {
            "good": "from cfspopcon.algorithm_class import Algorithm\n"
            "_probe_retry = Algorithm.from_single_function(lambda x: x, return_keys=['y'], name='_probe_retry', skip_unit_conversion=True)\n",
            "broken": "raise ImportError('broken on purpose')\n",
        },
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        with pytest.raises(ImportError, match="broken on purpose"):
            register_plugin("_probe_retry_pkg")

        (pkg / "broken.py").write_text("x = 1\n")
        importlib.invalidate_caches()
        register_plugin("_probe_retry_pkg")
        assert isinstance(Algorithm.get_algorithm("_probe_retry"), Algorithm)
    finally:
        forget_packages("_probe_retry_pkg")


def test_rollback_leaves_an_already_registered_package_alone(tmp_path, monkeypatch, clean_composites):
    """Only what the failing call imported is undone; an earlier registration survives."""
    write_package(
        tmp_path,
        "_probe_bystander_pkg",
        {
            "m": "from cfspopcon.algorithm_class import Algorithm\n"
            "_probe_bystander = Algorithm.from_single_function(lambda x: x, return_keys=['y'], name='_probe_bystander', skip_unit_conversion=True)\n"
        },
    )
    write_package(tmp_path, "_probe_faulty_pkg", {"m": "raise ImportError('broken on purpose')\n"})
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        register_plugin("_probe_bystander_pkg")
        with pytest.raises(ImportError, match="broken on purpose"):
            register_plugin("_probe_faulty_pkg")

        assert isinstance(Algorithm.get_algorithm("_probe_bystander"), Algorithm)
        assert "_probe_bystander_pkg" in sys.modules
    finally:
        forget_packages("_probe_bystander_pkg", "_probe_faulty_pkg")


def test_a_misspelled_algorithm_name_suggests_the_real_one():
    """A near-miss lookup suggests the registered name."""
    with pytest.raises(KeyError, match="Did you mean 'calc_plasma_volume'"):
        Algorithm.get_algorithm("calc_plasma_volme")


def test_the_popcon_command_discovers_before_reading_the_case(run_script):
    """popcon populates the registry before read_case runs, surfacing registration failures at startup.

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
    """popcon_algorithms writes the full listing, with the builtins registered by the time the file is written.

    Run in a subprocess: in-process this would pass whether the command discovers or not.
    """
    output = tmp_path / "algorithms.yaml"
    script = (
        "from click.testing import CliRunner\n"
        "from cfspopcon.cli import write_algorithms_yaml\n"
        f"result = CliRunner().invoke(write_algorithms_yaml, ['--output', {str(output)!r}])\n"
        # CliRunner captures the command's exception; include it in the failure message.
        "assert result.exit_code == 0, result.exception or result.output\n"
    )
    run_script(script)
    entries = [line for line in output.read_text().splitlines() if line and not line.startswith((" ", "#"))]
    assert len(entries) > 100, entries


# --- A package built on cfspopcon, extending the builtin registry ---------------------------------

DOWNSTREAM = "_ds_probe_pkg"

#: A builtin composite, named explicitly, so which one is under test stays the same whatever
#: order the walk registers them in.
BUILTIN_COMPOSITE = "calc_peaking_and_analytic_profiles"

#: Composites from a single algorithm through to one nested three deep, declared in a module the walk
#: reaches *before* the one defining the algorithm: none of their components exist at declaration.
DOWNSTREAM_DECLARATIONS = """\
from cfspopcon.algorithm_class import Algorithm, CompositeAlgorithm

# The point of declaring here: nothing this file names has been registered yet.
assert "calc_ds_metric" not in Algorithm.instances

_ds_own = CompositeAlgorithm.declare(["calc_ds_metric"], name="_ds_own")
_ds_mixed = CompositeAlgorithm.declare(["calc_plasma_volume", "calc_ds_metric"], name="_ds_mixed")
_ds_of_builtin_composite = CompositeAlgorithm.declare(["{BUILTIN_COMPOSITE}", "_ds_own"], name="_ds_of_builtin_composite")
_ds_deep = CompositeAlgorithm.declare(["_ds_of_builtin_composite", "_ds_mixed"], name="_ds_deep")
"""

#: A new variable of the package's own, so this also exercises declaring default units in code,
#: with ``extend_default_units_map``.
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
        register_plugin(DOWNSTREAM)
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
    """Algorithm objects compose and run directly, without the registry being touched at all.

    Run in a subprocess, to assert that the registry stays empty.
    """
    script = """
import xarray as xr
from cfspopcon.algorithm_class import Algorithm, CompositeAlgorithm
from cfspopcon.unit_handling import Quantity, extend_default_units_map, ureg

extend_default_units_map({"_solo_area": "m**2"})


@Algorithm.register_algorithm(return_keys=["_solo_area"])
def calc_solo_area(_solo_width, _solo_height):
    return _solo_width * _solo_height


@Algorithm.register_algorithm(return_keys=["_solo_label"], skip_unit_conversion=True)
def calc_solo_label(_solo_area):
    return "big" if _solo_area > Quantity(1.0, ureg.m**2) else "small"


# The decorator labels without registering, so the objects compose directly and the registry
# stays untouched.
composite = CompositeAlgorithm([calc_solo_area.__popcon_algorithm__, calc_solo_label.__popcon_algorithm__])
assert Algorithm.instances == {}, Algorithm.instances

inputs = xr.Dataset({"_solo_width": Quantity(2.0, ureg.m), "_solo_height": Quantity(3.0, ureg.m)})
assert composite.validate_inputs(inputs, quiet=True)
result = composite.update_dataset(inputs)
assert result["_solo_area"] == Quantity(6.0, ureg.m**2)
assert result["_solo_label"] == "big"
"""
    run_script(script)


def test_registering_a_package_registers_the_builtins_first(tmp_path, run_script):
    """register_plugin on a plugin brings the builtins in first, and importing alone brings nothing.

    Run in a subprocess, so the builtins are provably absent beforehand.
    """
    package = tmp_path / "_probe_explicit_pkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "algos.py").write_text(
        "from cfspopcon.algorithm_class import Algorithm\n"
        "_probe_explicit = Algorithm.from_single_function(lambda x: x, return_keys=['y'], name='_probe_explicit', skip_unit_conversion=True)\n"
    )
    script = (
        f"import sys; sys.path.insert(0, {str(tmp_path)!r})\n"
        "import cfspopcon\n"
        "import _probe_explicit_pkg.algos\n"
        "from cfspopcon import Algorithm\n"
        "assert Algorithm.instances == {}, 'importing must register nothing'\n"
        "cfspopcon.register_plugin('_probe_explicit_pkg')\n"
        "assert '_probe_explicit' in Algorithm.instances\n"
        "assert 'calc_plasma_volume' in Algorithm.instances, 'the builtins must be registered before a plugin'\n"
    )
    run_script(script)


def test_a_packages_own_variables_file_is_read_on_registration(tmp_path, monkeypatch, clean_composites):
    """A variables.yaml in the package root declares the default units of the package's variables."""
    from cfspopcon.unit_handling import default_units

    monkeypatch.setattr(default_units, "_DEFAULT_UNIT_BY_VARIABLE", dict(default_units._DEFAULT_UNIT_BY_VARIABLE))
    write_package(
        tmp_path,
        "_probe_units_pkg",
        {"m": "x = 1\n"},
    )
    (tmp_path / "_probe_units_pkg" / "variables.yaml").write_text(
        "_probe_units_var:\n  default_units: m**3\n  description:\n  - A probe variable.\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        register_plugin("_probe_units_pkg")
        assert default_units.default_unit("_probe_units_var") == "m**3"
    finally:
        forget_packages("_probe_units_pkg")


def test_changing_an_existing_variables_units_is_refused(tmp_path, monkeypatch, clean_composites):
    """A package redefining the default units of an existing variable is rejected, naming the variable."""
    from cfspopcon.unit_handling import default_units

    monkeypatch.setattr(default_units, "_DEFAULT_UNIT_BY_VARIABLE", dict(default_units._DEFAULT_UNIT_BY_VARIABLE))
    write_package(
        tmp_path,
        "_probe_clash_pkg",
        {"m": "from cfspopcon.unit_handling import extend_default_units_map\nextend_default_units_map({'average_electron_temp': 'eV'})\n"},
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        with pytest.raises(ValueError, match="average_electron_temp"):
            register_plugin("_probe_clash_pkg")
        # Re-declaring identical units is a no-op, so re-reading the same file is allowed.
        default_units.extend_default_units_map({"average_electron_temp": default_units.default_unit("average_electron_temp")})
    finally:
        forget_packages("_probe_clash_pkg")


def test_a_plain_module_is_refused_with_a_clear_error(tmp_path, monkeypatch):
    """A registration target must be a package; a single-file module is rejected by name."""
    (tmp_path / "_probe_plain_module.py").write_text("x = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        with pytest.raises(ValueError, match="_probe_plain_module.*must be a package"):
            register_plugin("_probe_plain_module")
    finally:
        forget_packages("_probe_plain_module")


def test_a_required_package_is_registered_first(tmp_path, monkeypatch, clean_composites):
    """__popcon_requires__ registers the named package before the declaring one, so composites may span them."""
    write_package(
        tmp_path,
        "_probe_req_dep_pkg",
        {
            "m": "from cfspopcon.algorithm_class import Algorithm\n"
            "_probe_req_dep = Algorithm.from_single_function(lambda x: x, return_keys=['y'], name='_probe_req_dep', skip_unit_conversion=True)\n"
        },
    )
    write_package(
        tmp_path,
        "_probe_req_main_pkg",
        {
            "__init__": "__popcon_requires__ = ('_probe_req_dep_pkg',)\n",
            "m": "from cfspopcon.algorithm_class import Algorithm, CompositeAlgorithm\n"
            "_probe_req_main = Algorithm.from_single_function(lambda x: x, return_keys=['y'], name='_probe_req_main', skip_unit_conversion=True)\n"
            "_probe_req_chain = CompositeAlgorithm.declare(['_probe_req_dep', '_probe_req_main'], name='_probe_req_chain')\n",
        },
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        register_plugin("_probe_req_main_pkg")
        assert isinstance(Algorithm.get_algorithm("_probe_req_dep"), Algorithm)
        assert isinstance(Algorithm.get_algorithm("_probe_req_chain"), CompositeAlgorithm)
    finally:
        forget_packages("_probe_req_dep_pkg", "_probe_req_main_pkg")


def test_circular_requirements_raise(tmp_path, monkeypatch, clean_composites):
    """Two packages requiring each other fail loudly, naming the cycle."""
    write_package(tmp_path, "_probe_cycle_a_pkg", {"__init__": "__popcon_requires__ = ('_probe_cycle_b_pkg',)\n"})
    write_package(tmp_path, "_probe_cycle_b_pkg", {"__init__": "__popcon_requires__ = ('_probe_cycle_a_pkg',)\n"})
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        with pytest.raises(RuntimeError, match="Circular __popcon_requires__"):
            register_plugin("_probe_cycle_a_pkg")
    finally:
        forget_packages("_probe_cycle_a_pkg", "_probe_cycle_b_pkg")
