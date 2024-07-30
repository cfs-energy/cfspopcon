from pathlib import Path

import pytest
import xarray as xr
from regression_results.generate_regression_results import (
    CASES_DIR,
    ALL_CASE_NAMES,
    ALL_CASE_PATHS,
)
from xarray.testing import assert_allclose

from cfspopcon.file_io import read_dataset_from_netcdf, write_dataset_to_netcdf
from cfspopcon.input_file_handling import read_case


@pytest.mark.regression
@pytest.mark.parametrize("case", ALL_CASE_PATHS, ids=ALL_CASE_NAMES)
@pytest.mark.filterwarnings("ignore:Not all input parameters were used")
def test_regression_against_case(case: Path):
    input_parameters, algorithm, _, _ = read_case(case)
    case_name = case.parent.stem

    dataset = algorithm.run(**input_parameters).merge(input_parameters)
    write_dataset_to_netcdf(dataset, Path(__file__).parent / "regression_results" / f"test1_{case.parent.stem}.nc")

    dataset = read_dataset_from_netcdf(Path(__file__).parent / "regression_results" / f"test1_{case.parent.stem}.nc").load()
    reference_dataset = read_dataset_from_netcdf(Path(__file__).parent / "regression_results" / f"{case_name}_result.nc").load()

    dataset, reference_dataset = xr.align(dataset, reference_dataset)
    assert_allclose(dataset, reference_dataset)


@pytest.mark.regression
@pytest.mark.parametrize("case", ALL_CASE_PATHS, ids=ALL_CASE_NAMES)
@pytest.mark.filterwarnings("ignore:Not all input parameters were used")
def test_regression_against_case_with_update(case: Path):
    input_parameters, algorithm, _, _ = read_case(case)
    case_name = case.parent.stem

    dataset = xr.Dataset(input_parameters)

    dataset = algorithm.update_dataset(dataset)
    write_dataset_to_netcdf(dataset, Path(__file__).parent / "regression_results" / f"test2_{case.parent.stem}.nc")

    dataset = read_dataset_from_netcdf(Path(__file__).parent / "regression_results" / f"test2_{case.parent.stem}.nc").load()
    reference_dataset = read_dataset_from_netcdf(Path(__file__).parent / "regression_results" / f"{case_name}_result.nc").load()

    dataset, reference_dataset = xr.align(dataset, reference_dataset)
    assert_allclose(dataset, reference_dataset)


@pytest.mark.regression
@pytest.mark.filterwarnings("ignore:Not all input parameters were used")
def test_regression_against_case_with_repeated_update():
    case = CASES_DIR / "SPARC_PRD" / "input.yaml"

    input_parameters, algorithm, _, _ = read_case(case)
    input_parameters["average_electron_density"] = input_parameters["average_electron_density"][::5]
    input_parameters["average_electron_temp"] = input_parameters["average_electron_temp"][::5]

    dataset = xr.Dataset(input_parameters)

    first_run = algorithm.update_dataset(dataset)
    second_run = algorithm.update_dataset(first_run)

    for variable in ["atomic_data"]:
        first_run = first_run.drop_vars(variable)
        second_run = second_run.drop_vars(variable)

    assert_allclose(
        first_run.transpose("dim_average_electron_density", "dim_average_electron_temp", "dim_species", "dim_rho"),
        second_run.transpose("dim_average_electron_density", "dim_average_electron_temp", "dim_species", "dim_rho"),
    )
