"""Tests for registering plugin packages through an input file's `plugins` section."""

import pytest
from click.testing import CliRunner
from utils.throwaway_packages import forget_packages, write_package

from cfspopcon.cli import write_algorithms_yaml
from cfspopcon.input_file_handling import process_input_dictionary
from cfspopcon.unit_handling import Quantity, default_units

PLUGIN_INIT = ""

PLUGIN_ALGORITHMS = (
    "from cfspopcon.algorithm_class import Algorithm\n"
    "from cfspopcon.unit_handling import Unitfull\n"
    "\n\n"
    "@Algorithm.register_algorithm(return_keys=['_probe_metric'])\n"
    "def calc_probe_metric(plasma_volume: Unitfull) -> Unitfull:\n"
    '    """Twice the plasma volume."""\n'
    "    return 2.0 * plasma_volume\n"
)

PLUGIN_VARIABLES = "_probe_metric:\n  default_units: m**3\n  description:\n  - Twice the plasma volume.\n"


@pytest.fixture(autouse=True)
def isolated_units_map(monkeypatch, tmp_path):
    """Give each test a copy of the default units map and an importable tmp_path."""
    monkeypatch.setattr(default_units, "_DEFAULT_UNIT_BY_VARIABLE", dict(default_units._DEFAULT_UNIT_BY_VARIABLE))
    monkeypatch.syspath_prepend(str(tmp_path))


def write_plugin(tmp_path, name):
    """Write a self-registering plugin package with one algorithm and a variables file."""
    write_package(tmp_path, name, {"__init__": PLUGIN_INIT, "algorithms": PLUGIN_ALGORITHMS})
    (tmp_path / name / "variables.yaml").write_text(PLUGIN_VARIABLES)
    return name


def test_an_input_file_can_name_a_plugin_algorithm(tmp_path, clean_composites):
    """The `plugins` section is registered before the algorithm names are resolved."""
    plugin = write_plugin(tmp_path, "_probe_case_plugin")
    try:
        repr_d, algorithm, _, _ = process_input_dictionary(
            {"plugins": [plugin], "algorithms": ["calc_probe_metric"], "plasma_volume": 3.0},
            case_dir=tmp_path,
        )
        assert algorithm(plasma_volume=repr_d["plasma_volume"]) == Quantity(6.0, "m**3")
    finally:
        forget_packages(plugin)


def test_a_single_plugin_may_be_given_as_a_string(tmp_path, clean_composites):
    """read_case's kwargs override passes strings, so a bare name must work like a one-element list."""
    plugin = write_plugin(tmp_path, "_probe_string_plugin")
    try:
        _, algorithm, _, _ = process_input_dictionary(
            {"plugins": plugin, "algorithms": ["calc_probe_metric"], "plasma_volume": 3.0},
            case_dir=tmp_path,
        )
        assert algorithm.name == "calc_probe_metric"
    finally:
        forget_packages(plugin)


def test_a_missing_plugin_is_blamed_on_the_input_file(tmp_path):
    """The error names the missing package and the section it came from."""
    with pytest.raises(ModuleNotFoundError, match=r"_probe_no_such_plugin.*'plugins' section"):
        process_input_dictionary({"plugins": ["_probe_no_such_plugin"], "algorithms": []}, case_dir=tmp_path)


def test_a_plugins_own_missing_dependency_is_not_blamed_on_the_input_file(tmp_path, clean_composites):
    """A missing import inside the plugin propagates unchanged, naming the actual missing module."""
    write_package(tmp_path, "_probe_dep_plugin", {"__init__": "import _probe_missing_dependency\n"})
    try:
        with pytest.raises(ModuleNotFoundError) as excinfo:
            process_input_dictionary({"plugins": ["_probe_dep_plugin"], "algorithms": []}, case_dir=tmp_path)
        assert excinfo.value.name == "_probe_missing_dependency"
        assert "'plugins' section" not in str(excinfo.value)
    finally:
        forget_packages("_probe_dep_plugin")


def test_the_algorithm_listing_can_include_a_plugin(tmp_path, clean_composites):
    """popcon_algorithms --plugin registers the plugin before writing the listing."""
    plugin = write_plugin(tmp_path, "_probe_listing_plugin")
    output = tmp_path / "algorithms.yaml"
    try:
        result = CliRunner().invoke(write_algorithms_yaml, ["-o", str(output), "--plugin", plugin])
        assert result.exit_code == 0, result.output
        assert "calc_probe_metric" in output.read_text()
    finally:
        forget_packages(plugin)
