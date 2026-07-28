import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import xarray as xr
import yaml

import cfspopcon
from cfspopcon import _discovery, discover_builtin_algorithms
from cfspopcon.algorithm_class import Algorithm, _pending_composites, build_pending_composites

xr.set_options(display_width=300)

REPOSITORY_ROOT = Path(cfspopcon.__file__).parents[1]


@pytest.fixture(scope="session", autouse=True)
def discovered_algorithms() -> None:
    """Populate the algorithm registry once for the suite, since discovery is explicit.

    Tests which need the registry as it is on a bare import assert that in a subprocess.
    """
    discover_builtin_algorithms()


@pytest.fixture()
def clean_composites():
    """Isolate a test's registrations and declarations from the rest of the session.

    Set the pending list aside rather than building or discarding whatever is on it, and undo
    whatever the test registers so it cannot depend on, or leak into, another test.
    """
    saved = _pending_composites[:]
    registered = set(Algorithm.instances)
    _pending_composites.clear()
    yield
    # Diff the registry rather than asking each test to declare what it added: a test which fails
    # partway still registered whatever it got to, and that must not leak into the next one.
    for name in set(Algorithm.instances) - registered:
        del Algorithm.instances[name]
    _pending_composites[:] = saved
    if saved:
        # Set aside while something else built the real declarations, so nothing will build these
        # now. Do it here, or they stay pending for the rest of the session.
        build_pending_composites()


@pytest.fixture()
def run_script():
    """Run a snippet in a fresh interpreter, so process-wide registry state can be asserted on.

    Defaults to running from the repository root, and reports the child's own output on failure
    rather than just its exit status.
    """

    def run(script: str, cwd: Path = REPOSITORY_ROOT) -> None:
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, cwd=cwd, check=False)
        assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"

    return run


@pytest.fixture()
def fake_entry_points(monkeypatch):
    """Install fake entry points, each loading to one of the given targets."""

    def install(*targets):
        eps = [SimpleNamespace(load=lambda target=target: target) for target in targets]
        monkeypatch.setattr(_discovery, "entry_points", lambda group: eps)

    return install


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
