import subprocess
import sys
from pathlib import Path

import pytest
import xarray as xr
import yaml

import cfspopcon
from cfspopcon import discover_builtin_algorithms
from cfspopcon.algorithm_class import Algorithm

xr.set_options(display_width=300)

REPOSITORY_ROOT = Path(cfspopcon.__file__).parents[1]


@pytest.fixture(scope="session", autouse=True)
def discovered_algorithms() -> None:
    """Populate the algorithm registry once for the suite, so no test depends on which one touches it first.

    Tests which need the registry as it is on a bare import assert that in a subprocess.
    """
    discover_builtin_algorithms()


@pytest.fixture()
def clean_composites():
    """Undo whatever the test registers, so it cannot depend on, or leak into, another test."""
    registered = set(Algorithm.instances)
    yield
    # Diffing the registry catches everything the test registered, including whatever a test
    # which failed partway got to, so nothing leaks into the next one.
    for name in set(Algorithm.instances) - registered:
        del Algorithm.instances[name]


@pytest.fixture()
def run_script():
    """Run a snippet in a fresh interpreter, so process-wide registry state can be asserted on.

    Defaults to running from the repository root, and includes the child's stdout and stderr in
    the failure message.
    """

    def run(script: str, cwd: Path = REPOSITORY_ROOT) -> None:
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, cwd=cwd, check=False)
        assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"

    return run


@pytest.fixture(scope="session")
def test_directory() -> Path:
    path = Path(__file__).parent
    assert path.exists()
    return path


@pytest.fixture(scope="session")
def repository_directory(test_directory) -> Path:
    path = test_directory.parent
    assert path.exists()
    return path


@pytest.fixture(scope="session")
def module_directory(repository_directory) -> Path:
    path = repository_directory / "cfspopcon"
    assert path.exists()
    return path


@pytest.fixture(scope="session")
def cases_directory(repository_directory) -> Path:
    path = repository_directory / "example_cases"
    assert path.exists()
    return path


@pytest.fixture(scope="session")
def example_inputs(cases_directory) -> dict:
    filepath = cases_directory / "SPARC_PRD" / "input.yaml"
    assert filepath.exists()

    return yaml.safe_load(filepath)
