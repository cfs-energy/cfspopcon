from pathlib import Path

import pytest
import xarray as xr
import yaml

xr.set_options(display_width=300)


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
