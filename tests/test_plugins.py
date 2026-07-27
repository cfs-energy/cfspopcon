"""Tests for cfspopcon.plugins.register_plugin."""

import sys
from pathlib import Path
from textwrap import dedent

import pytest

from cfspopcon import Algorithm, PluginClashError, register_plugin
from cfspopcon.plugins import _failed_registrations, _successful_registrations
from cfspopcon.unit_handling import ureg
from cfspopcon.unit_handling.default_units import default_unit, default_units_map, extend_default_units_map, reset_default_units

OK_PLUGIN = dedent(
    """
    from cfspopcon.algorithm_class import Algorithm
    from cfspopcon.unit_handling import Unitfull
    from cfspopcon.unit_handling.default_units import extend_default_units_map

    extend_default_units_map({"test_plugin_metric": "dimensionless"})

    @Algorithm.register_algorithm(return_keys=["test_plugin_metric"])
    def calc_test_plugin_metric(average_electron_density: Unitfull, greenwald_density_limit: Unitfull) -> Unitfull:
        return average_electron_density / greenwald_density_limit
    """
)

CLASH_PLUGIN = dedent(
    """
    from cfspopcon.algorithm_class import Algorithm
    from cfspopcon.unit_handling import Unitfull
    from cfspopcon.unit_handling.default_units import extend_default_units_map

    extend_default_units_map({"average_electron_density": "electron_volt"})

    @Algorithm.register_algorithm(return_keys=["average_electron_density"])
    def calc_clash_plugin_metric(major_radius: Unitfull) -> Unitfull:
        return major_radius
    """
)

BROKEN_PLUGIN = dedent(
    """
    from cfspopcon.unit_handling.default_units import extend_default_units_map

    extend_default_units_map({"test_broken_plugin_metric": "dimensionless"})
    raise ImportError("broken plugin")
    """
)


@pytest.fixture(autouse=True)
def restore_registries(monkeypatch, tmp_path):
    """Isolate the global registries and make tmp_path importable."""
    monkeypatch.syspath_prepend(str(tmp_path))
    algorithms_before = set(Algorithm.instances)
    units_before = default_units_map()
    modules_before = set(sys.modules)

    yield

    for name in set(Algorithm.instances) - algorithms_before:
        del Algorithm.instances[name]
    reset_default_units()
    extend_default_units_map(units_before)
    for name in set(sys.modules) - modules_before:
        del sys.modules[name]
    _successful_registrations.clear()
    _failed_registrations.clear()


def write_plugin(tmp_path: Path, name: str, source: str) -> str:
    """Write a plugin module into tmp_path and return its import name."""
    (tmp_path / f"{name}.py").write_text(source)
    return name


def test_register_plugin_reports_additions(tmp_path):
    """A well-behaved plugin registers and its report lists the additions."""
    name = write_plugin(tmp_path, "popcon_test_plugin_ok", OK_PLUGIN)

    report = register_plugin(name)

    assert report.algorithms == ["calc_test_plugin_metric"]
    assert report.variables == ["test_plugin_metric"]
    assert report.units == []
    assert default_unit("test_plugin_metric") == "dimensionless"
    assert "calc_test_plugin_metric" in str(Algorithm.get_algorithm("calc_test_plugin_metric").__doc__)


def test_registered_algorithm_runs_in_composite_chain(tmp_path):
    """A plugin algorithm composes and runs with built-in algorithms."""
    name = write_plugin(tmp_path, "popcon_test_plugin_chain", OK_PLUGIN)
    register_plugin(name)

    workflow = Algorithm.get_algorithm("calc_greenwald_density_limit") + Algorithm.get_algorithm("calc_test_plugin_metric")
    dataset = workflow.run(
        plasma_current=ureg.Quantity(8.7, "MA"),
        minor_radius=ureg.Quantity(0.57, "m"),
        average_electron_density=ureg.Quantity(25.0, "_1e19_per_cubic_metre"),
    )

    assert dataset["test_plugin_metric"].item() == pytest.approx(25.0 / 85.235, rel=1e-3)


def test_register_plugin_is_idempotent(tmp_path):
    """Registering the same plugin twice returns the original report."""
    name = write_plugin(tmp_path, "popcon_test_plugin_twice", OK_PLUGIN)

    report = register_plugin(name)

    assert register_plugin(name) is report


def test_clash_is_rejected_and_rolled_back(tmp_path):
    """A plugin redefining an existing variable is rejected and rolled back."""
    name = write_plugin(tmp_path, "popcon_test_plugin_clash", CLASH_PLUGIN)

    with pytest.raises(PluginClashError, match="average_electron_density"):
        register_plugin(name)

    assert default_unit("average_electron_density") == "_1e19_per_cubic_metre"
    assert "calc_clash_plugin_metric" not in Algorithm.instances

    # Repeated calls raise the same error instead of reporting an empty success.
    with pytest.raises(PluginClashError, match="average_electron_density"):
        register_plugin(name)


def test_failed_import_is_rolled_back(tmp_path):
    """A plugin that raises during import leaves no partial registrations."""
    name = write_plugin(tmp_path, "popcon_test_plugin_broken", BROKEN_PLUGIN)

    with pytest.raises(ImportError, match="broken plugin"):
        register_plugin(name)

    assert "test_broken_plugin_metric" not in default_units_map()
