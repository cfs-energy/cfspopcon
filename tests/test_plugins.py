"""Tests for plugin registration, both from Python and from an input file's `plugins` section."""

import sys
from pathlib import Path
from textwrap import dedent

import pytest
import xarray as xr
import yaml

from cfspopcon import Algorithm, process_input_dictionary, register_plugins
from cfspopcon.unit_handling import Quantity, ureg
from cfspopcon.unit_handling.default_units import default_unit, default_units_map, extend_default_units_map, reset_default_units

ALGORITHMS = dedent(
    """
    from cfspopcon.algorithm_class import Algorithm
    from cfspopcon.unit_handling import Unitfull

    @Algorithm.register_algorithm(return_keys=["plugin_metric"])
    def calc_plugin_metric(plasma_volume: Unitfull) -> Unitfull:
        return 2.0 * plasma_volume
    """
)

VARIABLES = """
plugin_metric:
  default_units: m**3
  description:
  - Twice the plasma volume.
"""


@pytest.fixture(autouse=True)
def restore_registries(monkeypatch, tmp_path):
    """Isolate the global registries and the import cache, and make tmp_path importable."""
    monkeypatch.syspath_prepend(str(tmp_path))
    algorithms_before = dict(Algorithm.instances)
    units_before = default_units_map()
    modules_before = set(sys.modules)

    yield

    Algorithm.instances.clear()
    Algorithm.instances.update(algorithms_before)
    reset_default_units()
    extend_default_units_map(units_before)
    for name in set(sys.modules) - modules_before:
        del sys.modules[name]


def write_plugin(tmp_path: Path, name: str, algorithms: str = ALGORITHMS, variables: str | None = VARIABLES) -> str:
    """Write a plugin package into tmp_path, and return its import name."""
    package = tmp_path / name
    package.mkdir()
    (package / "__init__.py").write_text("from . import algorithms\n")
    (package / "algorithms.py").write_text(algorithms)
    if variables is not None:
        (package / "plugin_variables.yaml").write_text(variables)
    return name


def test_a_plugin_registers_its_algorithms(tmp_path):
    """The algorithms are registered, their units come from the plugin's variables file, and they run."""
    registered = register_plugins(write_plugin(tmp_path, "popcon_plugin_ok"))

    assert registered == ["calc_plugin_metric"]
    assert default_unit("plugin_metric") == "m**3"

    workflow = Algorithm.get_algorithm("calc_plasma_volume") + Algorithm.get_algorithm("calc_plugin_metric")
    dataset = workflow.run(
        major_radius=Quantity(1.85, ureg.m),
        inverse_aspect_ratio=Quantity(0.3, ureg.dimensionless),
        areal_elongation=Quantity(1.75, ureg.dimensionless),
    )
    assert dataset["plugin_metric"] == 2.0 * dataset["plasma_volume"]


def test_a_plugin_without_a_variables_file_declares_its_own_units(tmp_path):
    """A plugin may call extend_default_units_map instead of shipping a variables file."""
    algorithms = 'from cfspopcon.unit_handling import extend_default_units_map\n\nextend_default_units_map({"plugin_metric": "m**3"})\n'
    register_plugins(write_plugin(tmp_path, "popcon_plugin_inline", algorithms=algorithms + ALGORITHMS, variables=None))

    assert default_unit("plugin_metric") == "m**3"


def test_a_plugin_which_does_not_import_is_rolled_back(tmp_path):
    """Nothing the plugin registered before it broke stays registered."""
    algorithms = ALGORITHMS + '\nraise ImportError("an optional dependency is missing")\n'
    name = write_plugin(tmp_path, "popcon_plugin_broken", algorithms=algorithms)

    with pytest.raises(ImportError, match="optional dependency"):
        register_plugins(name)

    assert "calc_plugin_metric" not in Algorithm.instances
    assert "plugin_metric" not in default_units_map()


def test_a_plugin_redefining_a_variables_units_is_rejected(tmp_path):
    """Redefining units already in the map would silently reinterpret every value of that variable."""
    name = write_plugin(tmp_path, "popcon_plugin_units_clash", variables="plasma_volume:\n  default_units: m**2\n")
    # An equivalent spelling is not a redefinition: "meter ** 3" would be accepted here.

    with pytest.raises(ValueError, match=r"redefine the default units of \[plasma_volume\]"):
        register_plugins(name)

    assert default_unit("plasma_volume") == "meter ** 3"
    assert "calc_plugin_metric" not in Algorithm.instances


def test_a_plugin_redefining_an_algorithm_is_rolled_back(tmp_path):
    """An algorithm name is already refused if it is taken; the rest of the plugin goes with it."""
    algorithms = ALGORITHMS.replace('return_keys=["plugin_metric"]', 'return_keys=["plugin_metric"], name="calc_plasma_volume"')
    name = write_plugin(tmp_path, "popcon_plugin_algorithm_clash", algorithms=algorithms)
    builtin = Algorithm.get_algorithm("calc_plasma_volume")

    with pytest.raises(RuntimeError, match="calc_plasma_volume"):
        register_plugins(name)

    assert Algorithm.get_algorithm("calc_plasma_volume") is builtin
    assert "plugin_metric" not in default_units_map()


def test_several_plugins_are_registered_as_one_set(tmp_path):
    """A failure in the second plugin takes the first one's registrations with it."""
    first = write_plugin(tmp_path, "popcon_plugin_first")
    second = write_plugin(tmp_path, "popcon_plugin_second", algorithms='raise ImportError("broken")\n', variables=None)

    with pytest.raises(ImportError, match="broken"):
        register_plugins(first, second)

    assert "calc_plugin_metric" not in Algorithm.instances


def test_an_input_file_can_name_a_plugin_algorithm(tmp_path):
    """The `plugins` section is registered before the algorithm names are resolved."""
    name = write_plugin(tmp_path, "popcon_plugin_case")
    case = dict(
        plugins=[name],
        algorithms=["calc_plasma_volume", "calc_plugin_metric"],
        major_radius=1.85,
        inverse_aspect_ratio=0.3,
        areal_elongation=1.75,
    )

    inputs, algorithm, _, _ = process_input_dictionary(yaml.safe_load(yaml.safe_dump(case)), tmp_path)
    dataset = algorithm.update_dataset(xr.Dataset(inputs))

    assert dataset["plugin_metric"] == 2.0 * dataset["plasma_volume"]


def test_an_input_file_naming_a_plugin_which_does_not_import_says_so(tmp_path):
    """The error names the missing package and the section it came from."""
    with pytest.raises(ModuleNotFoundError, match="'popcon_plugin_absent', listed in the 'plugins' section"):
        process_input_dictionary(dict(plugins=["popcon_plugin_absent"], algorithms=[]), tmp_path)
