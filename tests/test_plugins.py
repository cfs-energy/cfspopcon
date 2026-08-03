"""Tests for plugin registration, both from Python and from an input file's `plugins` section."""

import sys
from pathlib import Path
from textwrap import dedent

import pytest
import xarray as xr
import yaml
from utils.throwaway_packages import forget_packages, write_package

import cfspopcon
from cfspopcon import (
    Algorithm,
    CompositeAlgorithm,
    PluginClashError,
    algorithms_setting,
    algorithms_using,
    register_plugin,
    register_plugins,
)
from cfspopcon.algorithm_class import _pending_composites, pending_composites, restore_registry, registered_algorithms
from cfspopcon.plugins import _failed_registrations, _successful_registrations
from cfspopcon.unit_handling import Quantity, ureg
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
    algorithms_before = registered_algorithms()
    pending_before = pending_composites()
    units_before = default_units_map()
    modules_before = set(sys.modules)

    yield

    # Restore rather than diff-and-delete: a plugin may have replaced an entry as well as added one,
    # and a declaration left pending would make the next build raise about a package that has gone.
    restore_registry(algorithms_before, pending_before)
    reset_default_units()
    extend_default_units_map(units_before)
    forget_packages(*{name.split(".")[0] for name in set(sys.modules) - modules_before})
    _successful_registrations.clear()
    _failed_registrations.clear()


def write_plugin(tmp_path: Path, name: str, source: str, module: str = "algorithms") -> str:
    """Write a single-module plugin package into tmp_path and return its import name."""
    write_package(tmp_path, name, {module: source})
    return name


def test_register_plugin_reports_additions(tmp_path):
    """A well-behaved plugin registers and its report lists the additions."""
    name = write_plugin(tmp_path, "popcon_test_plugin_ok", OK_PLUGIN)

    report = register_plugin(name)

    assert report.algorithms == ("calc_test_plugin_metric",)
    assert report.variables == ("test_plugin_metric",)
    assert report.units == ()
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


def test_plugin_algorithms_appear_in_registry_queries(tmp_path):
    """The registry queries include a plugin's algorithms once it is registered."""
    name = write_plugin(tmp_path, "popcon_test_plugin_queries", OK_PLUGIN)
    register_plugin(name)

    assert "calc_test_plugin_metric" in algorithms_setting("test_plugin_metric")
    assert "calc_test_plugin_metric" in algorithms_using("average_electron_density")


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


# --- The four fixes registration needed for the discovery model -----------------------------------

DISCOVERS_THEN_FAILS = dedent(
    """
    import cfspopcon
    cfspopcon.discover_builtin_algorithms()   # a plugin naming builtins in a composite would do this
    raise ImportError("fails after discovering")
    """
)

SUBMODULE_PLUGIN = dedent(
    """
    from cfspopcon.algorithm_class import Algorithm
    from cfspopcon.unit_handling import Unitfull
    from cfspopcon.unit_handling.default_units import extend_default_units_map

    extend_default_units_map({"deep_plugin_metric": "meter"})

    @Algorithm.register_algorithm(return_keys=["deep_plugin_metric"])
    def calc_deep_plugin_metric(major_radius: Unitfull) -> Unitfull:
        return major_radius
    """
)

COMPOSITE_PLUGIN = dedent(
    """
    from cfspopcon.algorithm_class import Algorithm, CompositeAlgorithm
    from cfspopcon.unit_handling import Unitfull
    from cfspopcon.unit_handling.default_units import extend_default_units_map

    extend_default_units_map({"composite_plugin_metric": "meter**3"})

    @Algorithm.register_algorithm(return_keys=["composite_plugin_metric"])
    def calc_composite_plugin_metric(plasma_volume: Unitfull) -> Unitfull:
        return 2.0 * plasma_volume

    CompositeAlgorithm.register_from_list(
        ["calc_plasma_volume", "calc_composite_plugin_metric"], name="plugin_composite"
    )
    """
)

UNBUILDABLE_COMPOSITE_PLUGIN = dedent(
    """
    from cfspopcon.algorithm_class import CompositeAlgorithm

    CompositeAlgorithm.register_from_list(["no_such_algorithm"], name="unbuildable_composite")
    """
)

SHRINKING_PLUGIN = dedent(
    """
    from cfspopcon.unit_handling.default_units import extend_default_units_map, reset_default_units

    # reset_default_units empties the map rather than restoring the builtins, so this loses every
    # variable cfspopcon defines. Removing a variable's units is as damaging as changing them.
    reset_default_units()
    extend_default_units_map({"shrinking_plugin_metric": "meter"})
    """
)

OVERRIDE_PLUGIN = dedent(
    """
    from cfspopcon.algorithm_class import Algorithm
    from cfspopcon.unit_handling import Unitfull

    @Algorithm.register_algorithm(return_keys=["plasma_volume"], name="calc_plasma_volume", override=True)
    def sneaky_replacement(major_radius: Unitfull) -> Unitfull:
        return major_radius
    """
)


def test_a_plugin_which_discovers_the_builtins_does_not_take_them_down_with_it(tmp_path, run_script):
    """Rollback must not delete the builtins, which it would if they landed inside the diff.

    Registration has to discover them before snapshotting. Run in a subprocess with an un-discovered
    registry: in-process the suite has already discovered, so the builtins are inside the snapshot
    whether registration puts them there or not, and this would pass either way. The failure is also
    unrecoverable -- the walk only imports, and the modules are already imported -- so a later
    discover_builtin_algorithms() cannot put them back.
    """
    name = write_plugin(tmp_path, "popcon_test_plugin_discovers", DISCOVERS_THEN_FAILS)
    script = (
        "import sys\n"
        f"sys.path.insert(0, {str(tmp_path)!r})\n"
        "import cfspopcon\n"
        "from cfspopcon import Algorithm\n"
        "assert Algorithm.instances == {}, 'expected an un-discovered registry'\n"
        "try:\n"
        f"    cfspopcon.register_plugin({name!r})\n"
        "except ImportError:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError('the plugin should have failed to import')\n"
        "cfspopcon.discover_builtin_algorithms()   # cannot help: the modules are already imported\n"
        "assert len(Algorithm.instances) > 100, len(Algorithm.instances)\n"
    )
    run_script(script)


def test_a_plugin_package_is_walked_not_merely_imported(tmp_path):
    """An algorithm in a plugin submodule is registered, so a plugin may be more than one module."""
    write_package(tmp_path, "popcon_test_plugin_deep", {"models/detachment": SUBMODULE_PLUGIN})

    report = register_plugin("popcon_test_plugin_deep")

    assert report.algorithms == ("calc_deep_plugin_metric",)
    assert isinstance(Algorithm.get_algorithm("calc_deep_plugin_metric"), Algorithm)


def test_a_plugin_composite_is_built_and_reported(tmp_path):
    """register_from_list is the documented way to declare a composite, so it must be built."""
    name = write_plugin(tmp_path, "popcon_test_plugin_composite", COMPOSITE_PLUGIN)

    report = register_plugin(name)

    assert isinstance(Algorithm.get_algorithm("plugin_composite"), CompositeAlgorithm)
    assert report.algorithms == ("calc_composite_plugin_metric", "plugin_composite")
    assert not _pending_composites


def test_a_rejected_plugin_leaves_no_declaration_pending(tmp_path):
    """A declaration surviving rollback would make the next build raise about a plugin that has gone.

    The composite here names an algorithm nobody registers, so it is still pending when the walk's
    own build fails -- which is the only moment at which rollback has a declaration to undo.
    """
    name = write_plugin(tmp_path, "popcon_test_plugin_unbuildable", UNBUILDABLE_COMPOSITE_PLUGIN)
    pending_at_start = pending_composites()

    with pytest.raises(RuntimeError, match="no_such_algorithm"):
        register_plugin(name)

    assert pending_composites() == pending_at_start
    cfspopcon.algorithm_class.build_pending_composites()  # must not raise about the rejected plugin


def test_a_plugin_which_removes_a_variables_units_is_a_clash(tmp_path):
    """Losing a variable's default units is a clash as much as changing them is."""
    name = write_plugin(tmp_path, "popcon_test_plugin_shrinks", SHRINKING_PLUGIN)
    cfspopcon.discover_builtin_algorithms()

    with pytest.raises(PluginClashError, match=r"major_radius.*\(removed\)"):
        register_plugin(name)

    assert default_unit("major_radius") == "meter"


def test_an_override_of_a_builtin_is_a_clash(tmp_path):
    """override=True keeps the name, so only comparing names would let a replaced builtin through."""
    name = write_plugin(tmp_path, "popcon_test_plugin_override", OVERRIDE_PLUGIN)
    cfspopcon.discover_builtin_algorithms()
    original = Algorithm.get_algorithm("calc_plasma_volume")

    with pytest.raises(PluginClashError, match="algorithm 'calc_plasma_volume' was replaced"):
        register_plugin(name)

    assert Algorithm.get_algorithm("calc_plasma_volume") is original


# --- Several plugins as one unit ------------------------------------------------------------------

SPANNING_A = dedent(
    """
    from cfspopcon.algorithm_class import Algorithm, CompositeAlgorithm
    from cfspopcon.unit_handling import Unitfull
    from cfspopcon.unit_handling.default_units import extend_default_units_map

    extend_default_units_map({"span_a_metric": "meter"})

    @Algorithm.register_algorithm(return_keys=["span_a_metric"])
    def calc_span_a_metric(major_radius: Unitfull) -> Unitfull:
        return major_radius

    # Names an algorithm from the other plugin, which has not been walked yet.
    CompositeAlgorithm.register_from_list(["calc_span_a_metric", "calc_span_b_metric"], name="spanning_composite")
    """
)

SPANNING_B = dedent(
    """
    from cfspopcon.algorithm_class import Algorithm
    from cfspopcon.unit_handling import Unitfull
    from cfspopcon.unit_handling.default_units import extend_default_units_map

    extend_default_units_map({"span_b_metric": "meter"})

    @Algorithm.register_algorithm(return_keys=["span_b_metric"])
    def calc_span_b_metric(minor_radius: Unitfull) -> Unitfull:
        return minor_radius
    """
)


@pytest.fixture()
def spanning_plugins(tmp_path):
    """Two plugins whose composite spans them both, declared in the one walked first."""
    write_plugin(tmp_path, "popcon_test_span_a", SPANNING_A)
    write_plugin(tmp_path, "popcon_test_span_b", SPANNING_B)
    return "popcon_test_span_a", "popcon_test_span_b"


def test_a_composite_may_span_plugins_registered_together(spanning_plugins):
    """One outer scope for the set, so the composite build waits for every plugin."""
    reports = register_plugins(*spanning_plugins)

    assert isinstance(Algorithm.get_algorithm("spanning_composite"), CompositeAlgorithm)
    # Attributed to the plugin that declared it, not to the one that completed it.
    assert reports[0].algorithms == ("calc_span_a_metric", "spanning_composite")
    assert reports[1].algorithms == ("calc_span_b_metric",)


def test_registering_the_same_plugins_one_at_a_time_cannot_span_them(spanning_plugins):
    """The naive loop: each call is its own outer scope, so the first build has nothing to build on."""
    first, second = spanning_plugins

    with pytest.raises(RuntimeError, match="calc_span_b_metric"):
        register_plugin(first)

    # And the failure aborted before the second plugin, so nothing registered it either.
    assert "calc_span_b_metric" not in Algorithm.instances


def test_a_failing_plugin_rolls_back_the_whole_set(tmp_path):
    """All-or-nothing: a plugin listed before the failure must not stay half-registered."""
    good = write_plugin(tmp_path, "popcon_test_set_good", SUBMODULE_PLUGIN)
    broken = write_plugin(tmp_path, "popcon_test_set_broken", BROKEN_PLUGIN)

    with pytest.raises(ImportError, match="broken plugin"):
        register_plugins(good, broken)

    assert "calc_deep_plugin_metric" not in Algorithm.instances
    assert "deep_plugin_metric" not in default_units_map()
    assert _successful_registrations == {}


# --- The `plugins` section of an input file --------------------------------------------------------


def write_case(tmp_path: Path, plugins: list[str], algorithms: list[str], **inputs) -> Path:
    """Write a case directory whose input.yaml lists the given plugins and algorithms."""
    case_dir = tmp_path / "case"
    case_dir.mkdir(exist_ok=True)
    (case_dir / "input.yaml").write_text(yaml.safe_dump({"plugins": plugins, "algorithms": algorithms, **inputs}))
    return case_dir


def test_an_input_file_can_name_a_plugin_algorithm(tmp_path):
    """The point of the feature: a case file lists its own plugins and then uses their algorithms."""
    name = write_plugin(tmp_path, "popcon_test_case_plugin", COMPOSITE_PLUGIN)
    case_dir = write_case(
        tmp_path,
        plugins=[name],
        algorithms=["calc_plasma_volume", "calc_composite_plugin_metric"],
        major_radius=1.85,
        inverse_aspect_ratio=0.3,
        areal_elongation=1.75,
    )

    input_parameters, algorithm, _, _ = cfspopcon.read_case(case_dir)

    assert "plugins" not in input_parameters  # consumed, not treated as an input variable
    result = algorithm.update_dataset(xr.Dataset(input_parameters))
    assert result["composite_plugin_metric"].pint.units == ureg.m**3


def test_an_input_file_composite_may_span_its_plugins(tmp_path, spanning_plugins):
    """The `plugins` list is registered as one unit, so a spanning composite may be named."""
    case_dir = write_case(tmp_path, plugins=list(spanning_plugins), algorithms=["spanning_composite"])

    _, algorithm, _, _ = cfspopcon.read_case(case_dir)

    assert [alg._name for alg in algorithm.algorithms] == ["calc_span_a_metric", "calc_span_b_metric"]


def test_a_plugin_which_does_not_import_names_itself_and_the_input_file(tmp_path):
    """A bare ModuleNotFoundError does not say where the name came from."""
    case_dir = write_case(tmp_path, plugins=["popcon_test_no_such_plugin"], algorithms=[])

    with pytest.raises(ModuleNotFoundError, match="popcon_test_no_such_plugin.*'plugins' section"):
        cfspopcon.read_case(case_dir)


def test_a_case_without_a_plugins_section_is_unaffected(tmp_path):
    """The key is optional, and absent it nothing is imported."""
    case_dir = tmp_path / "case_no_plugins"
    case_dir.mkdir()
    (case_dir / "input.yaml").write_text(yaml.safe_dump({"algorithms": ["calc_plasma_volume"], "major_radius": 1.85}))

    input_parameters, algorithm, _, _ = cfspopcon.read_case(case_dir)

    assert algorithm._name == "calc_plasma_volume"
    assert "plugins" not in input_parameters


def test_the_popcon_command_runs_a_case_with_plugins(tmp_path, run_script):
    """popcon must register an input file's plugins for itself.

    Run in a subprocess: this suite discovers at session start and other tests have already imported
    plugin packages, so in-process this would pass whether the mechanism worked or not.
    """
    name = write_plugin(tmp_path, "popcon_test_cli_plugin", COMPOSITE_PLUGIN)
    case_dir = write_case(
        tmp_path,
        plugins=[name],
        algorithms=["calc_plasma_volume", "calc_composite_plugin_metric"],
        major_radius=1.85,
        inverse_aspect_ratio=0.3,
        areal_elongation=1.75,
    )
    script = (
        "import sys\n"
        f"sys.path.insert(0, {str(tmp_path)!r})\n"
        "import xarray as xr\n"
        "import cfspopcon.cli as cli\n"
        "from cfspopcon import Algorithm\n"
        "def stop_after_read_case(*args, **kwargs):\n"
        "    inputs, algorithm, points, plots = real_read_case(*args, **kwargs)\n"
        "    result = algorithm.update_dataset(xr.Dataset(inputs))\n"
        "    assert 'composite_plugin_metric' in result, list(result)\n"
        "    raise SystemExit(0)\n"
        "real_read_case = cli.read_case\n"
        "cli.read_case = stop_after_read_case\n"
        f"cli.run_popcon({str(case_dir)!r}, False, {{}})\n"
    )
    run_script(script)


# --- A plugin's own variables file -----------------------------------------------------------------

PLUGIN_VARIABLES = """\
file_declared_metric:
  default_units: meter**2
  description:
  - An area declared by the plugin's own variables file
  set_by:
  - calc_file_declared_metric
  used_by: []
file_declared_selector:
  default_units: null
  description:
  - A class-typed switch, declared with no units
"""

FILE_UNITS_PLUGIN = dedent(
    """
    from cfspopcon.algorithm_class import Algorithm
    from cfspopcon.unit_handling import Unitfull

    # No extend_default_units_map call: the units come from plugin_variables.yaml.
    @Algorithm.register_algorithm(return_keys=["file_declared_metric"])
    def calc_file_declared_metric(major_radius: Unitfull) -> Unitfull:
        return major_radius**2
    """
)


def test_a_plugin_declares_its_units_in_its_own_variables_file(tmp_path):
    """A plugin ships plugin_variables.yaml instead of writing a units dict in Python."""
    write_package(tmp_path, "popcon_test_plugin_file_units", {"algorithms": FILE_UNITS_PLUGIN})
    (tmp_path / "popcon_test_plugin_file_units" / "plugin_variables.yaml").write_text(PLUGIN_VARIABLES)

    report = register_plugin("popcon_test_plugin_file_units")

    assert default_unit("file_declared_metric") == "meter**2"
    assert default_unit("file_declared_selector") is None
    # The file's variables show up in the report the same way an inline declaration would.
    assert "file_declared_metric" in report.variables

    result = Algorithm.get_algorithm("calc_file_declared_metric").update_dataset(xr.Dataset({"major_radius": Quantity(2.0, ureg.m)}))
    assert result["file_declared_metric"] == Quantity(4.0, ureg.m**2)


def test_cfspopcons_own_variables_file_is_never_written_to(tmp_path):
    """A plugin's variables live in the plugin; cfspopcon's variables.yaml is left alone."""
    cfspopcon_variables = Path(cfspopcon.__file__).parent / "variables.yaml"
    before = cfspopcon_variables.read_bytes()

    write_package(tmp_path, "popcon_test_plugin_untouched", {"algorithms": FILE_UNITS_PLUGIN})
    (tmp_path / "popcon_test_plugin_untouched" / "plugin_variables.yaml").write_text(PLUGIN_VARIABLES)
    register_plugin("popcon_test_plugin_untouched")

    assert cfspopcon_variables.read_bytes() == before
    assert "file_declared_metric" not in yaml.safe_load(before)


def test_a_plugin_without_a_variables_file_is_unaffected(tmp_path):
    """The file is optional; a plugin declaring its units inline still works."""
    name = write_plugin(tmp_path, "popcon_test_plugin_no_file", SUBMODULE_PLUGIN)

    report = register_plugin(name)

    assert report.variables == ("deep_plugin_metric",)


# --- Which plugin in a rejected set is blamed ------------------------------------------------------


def test_only_the_offender_is_blamed_for_a_clash(tmp_path):
    """A plugin rolled back alongside an offender must not be accused of the offender's clash."""
    innocent = write_plugin(tmp_path, "popcon_test_innocent", SUBMODULE_PLUGIN)
    offender = write_plugin(tmp_path, "popcon_test_offender", CLASH_PLUGIN)

    with pytest.raises(PluginClashError) as raised:
        register_plugins(innocent, offender)

    # The raised error names the offender and its clash, and not the innocent plugin.
    assert "popcon_test_offender" in str(raised.value)
    assert "popcon_test_innocent" not in str(raised.value)

    # Asked again, each is told its own story.
    with pytest.raises(PluginClashError, match="average_electron_density"):
        register_plugin(offender)
    with pytest.raises(PluginClashError, match="registered nothing.*rolled back as part of a set"):
        register_plugin(innocent)


def test_a_plugin_redefining_an_earlier_plugins_variable_is_the_offender(tmp_path):
    """Clashes are diffed per plugin, so a plugin-versus-plugin redefinition is attributed correctly."""
    first = write_plugin(tmp_path, "popcon_test_first", SUBMODULE_PLUGIN)
    second = write_plugin(
        tmp_path,
        "popcon_test_second",
        dedent(
            """
            from cfspopcon.unit_handling.default_units import extend_default_units_map

            extend_default_units_map({"deep_plugin_metric": "second"})   # the other plugin's variable
            """
        ),
    )

    with pytest.raises(PluginClashError, match="popcon_test_second") as raised:
        register_plugins(first, second)

    assert "deep_plugin_metric" in str(raised.value)
    assert "popcon_test_first" not in str(raised.value)
