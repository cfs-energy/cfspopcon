"""A package built on cfspopcon can extend the registry, and can use it without the builtins.

Two applications are covered. First, a downstream package registering its own algorithms and
declaring composites which mix them with cfspopcon's, to whatever nesting depth. Second, a package
using the Algorithm/CompositeAlgorithm machinery on its own, having never called
discover_builtin_algorithms.
"""

import importlib

import pytest
import xarray as xr
from utils.throwaway_packages import forget_packages, write_package

import cfspopcon
from cfspopcon.algorithm_class import Algorithm, CompositeAlgorithm
from cfspopcon.unit_handling import Quantity, ureg

PACKAGE = "_ds_probe_pkg"

#: A builtin composite, named rather than picked out of the registry, so which one is under test does
#: not depend on the order the walk happened to register them in.
BUILTIN_COMPOSITE = "calc_peaking_and_analytic_profiles"

#: The composites the package declares, from a single algorithm through to one nested three deep.
#: Declaring them all in a module the walk reaches *before* the one defining the algorithm also
#: covers the out-of-order case: none of their components exist when they are declared.
DECLARATIONS = """\
from cfspopcon.algorithm_class import Algorithm, CompositeAlgorithm

# The point of declaring here: nothing this file names has been registered yet.
assert "calc_ds_metric" not in Algorithm.instances

CompositeAlgorithm.register_from_list(["calc_ds_metric"], name="_ds_own")
CompositeAlgorithm.register_from_list(["calc_plasma_volume", "calc_cylindrical_edge_safety_factor"], name="_ds_builtins")
CompositeAlgorithm.register_from_list(["calc_plasma_volume", "calc_ds_metric"], name="_ds_mixed")
CompositeAlgorithm.register_from_list(["{BUILTIN_COMPOSITE}", "_ds_own"], name="_ds_of_builtin_composite")
CompositeAlgorithm.register_from_list(["_ds_of_builtin_composite", "_ds_mixed"], name="_ds_deep")
"""

#: A new variable of the package's own, so this also exercises where a downstream package declares
#: default units: extend_default_units_map, since read_default_units_from_file takes no path.
ALGORITHMS = """\
from cfspopcon.algorithm_class import Algorithm
from cfspopcon.unit_handling import extend_default_units_map

extend_default_units_map({"_ds_metric": "m**3"})


@Algorithm.register_algorithm(return_keys=["_ds_metric"])
def calc_ds_metric(plasma_volume):
    \"\"\"Throwaway algorithm consuming a builtin algorithm's output.\"\"\"
    return 2.0 * plasma_volume
"""

DECLARED_COMPOSITES = ["_ds_own", "_ds_builtins", "_ds_mixed", "_ds_of_builtin_composite", "_ds_deep"]

PLASMA_VOLUME_INPUTS = {
    "major_radius": Quantity(1.85, ureg.m),
    "inverse_aspect_ratio": Quantity(0.3, ureg.dimensionless),
    "areal_elongation": Quantity(1.75, ureg.dimensionless),
}


@pytest.fixture()
def downstream_package(tmp_path, monkeypatch, clean_composites):
    """Write an importable package which extends the builtin registry, and clean up after it.

    Only writes it: each test chooses how the package gets loaded.
    """
    assert isinstance(Algorithm.get_algorithm(BUILTIN_COMPOSITE), CompositeAlgorithm)
    write_package(
        tmp_path,
        PACKAGE,
        {
            "a_declarations": DECLARATIONS.format(BUILTIN_COMPOSITE=BUILTIN_COMPOSITE),
            "z_algorithms": ALGORITHMS,
        },
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        yield tmp_path
    finally:
        forget_packages(PACKAGE)


def test_a_walk_extends_the_builtin_registry(downstream_package):
    """Every declared composite builds, including ones nested three deep across both packages."""
    cfspopcon.discover_algorithms_in_package(PACKAGE)

    for name in DECLARED_COMPOSITES:
        assert isinstance(Algorithm.get_algorithm(name), CompositeAlgorithm), name

    # Flattening reaches through both packages: the deepest composite runs the builtin composite's
    # algorithms as well as the new one.
    deep = Algorithm.get_algorithm("_ds_deep")
    member_names = [alg._name for alg in deep.algorithms]
    assert "calc_ds_metric" in member_names
    assert "calc_plasma_volume" in member_names
    builtin_members = [alg._name for alg in Algorithm.get_algorithm(BUILTIN_COMPOSITE).algorithms]
    assert set(builtin_members) <= set(member_names)


def test_a_downstream_composite_runs_end_to_end(downstream_package):
    """A composite mixing builtin and new algorithms computes the new variable."""
    cfspopcon.discover_algorithms_in_package(PACKAGE)

    dataset = xr.Dataset(PLASMA_VOLUME_INPUTS)
    result = Algorithm.get_algorithm("_ds_mixed").update_dataset(dataset)

    assert result["_ds_metric"] == 2.0 * result["plasma_volume"]
    assert result["_ds_metric"].pint.units == ureg.m**3


@pytest.mark.parametrize("target", ["module", "callable"])
def test_an_entry_point_provider_extends_the_registry(downstream_package, target, fake_entry_points, tmp_path):
    """Both entry-point target kinds register the package's algorithms during discovery.

    The target must not be resolved before discovery runs, or the registration would happen at
    resolution time and the test would pass whether the entry point was consulted or not.
    """
    if target == "module":
        # A module target is imported, not walked, so point it at a module which walks on import.
        write_package(tmp_path, "_ds_entry_pkg", {"__init__": f"import cfspopcon\ncfspopcon.discover_algorithms_in_package('{PACKAGE}')\n"})
        load = lambda: importlib.import_module("_ds_entry_pkg")  # noqa: E731
    else:
        load = lambda: lambda: cfspopcon.discover_algorithms_in_package(PACKAGE)  # noqa: E731

    fake_entry_points(loaders=[load])
    assert "calc_ds_metric" not in Algorithm.instances, "the entry point must not have been loaded yet"

    try:
        cfspopcon.discover_builtin_algorithms()

        for name in DECLARED_COMPOSITES:
            assert isinstance(Algorithm.get_algorithm(name), CompositeAlgorithm), name
    finally:
        # Must happen even if an assertion fails, or the module stays in sys.modules and the next
        # test to write over that name imports this one's copy.
        forget_packages("_ds_entry_pkg")


def test_walking_before_the_builtins_are_discovered_says_what_is_missing(downstream_package, run_script):
    """The composites name builtin algorithms, so discovery order matters and must be explained.

    Run in a subprocess: the suite has already discovered the builtins in this process.
    """
    script = (
        "import cfspopcon\n"
        f"try:\n    cfspopcon.discover_algorithms_in_package({PACKAGE!r})\n"
        "except RuntimeError as exc:\n"
        "    assert 'calc_plasma_volume' in str(exc), exc\n"
        "else:\n"
        "    raise AssertionError('should have failed without the builtins')\n"
    )
    # Run from where the package was written, so the subprocess can import it.
    run_script(script, cwd=downstream_package)


def test_the_machinery_works_without_the_builtin_algorithms(run_script):
    """A package may use Algorithm/CompositeAlgorithm without ever discovering cfspopcon's own.

    Covers both unit paths -- skip_unit_conversion, and a variable given units of its own -- and
    asserts the registry holds nothing but the package's own algorithms, so the test would fail if
    anything quietly pulled the builtins in. Run in a subprocess for that last assertion.
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
