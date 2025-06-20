import copy
from pathlib import Path

import pytest
import xarray as xr
from utils.regression_results import (
    ALL_CASE_NAMES,
    ALL_CASE_PATHS,
    CASES_DIR,
)
from xarray.testing import assert_allclose

from cfspopcon.file_io import read_dataset_from_netcdf, write_dataset_to_netcdf
from cfspopcon.input_file_handling import read_case

ignored_variables = ["radas_version"]


@pytest.mark.regression
@pytest.mark.parametrize("case", ALL_CASE_PATHS, ids=ALL_CASE_NAMES)
@pytest.mark.filterwarnings("ignore:Not all input parameters were used")
def test_regression_against_case(case: Path):
    input_parameters, algorithm, _, _ = read_case(case)
    case_name = case.parent.stem

    dataset = algorithm.run(**input_parameters).merge(input_parameters)
    write_dataset_to_netcdf(dataset, Path(__file__).parent / "regression_results" / f"test1_{case.parent.stem}.nc")

    dataset = (
        read_dataset_from_netcdf(Path(__file__).parent / "regression_results" / f"test1_{case.parent.stem}.nc")
        .load()
        .drop_vars(ignored_variables, errors="ignore")
    )
    reference_dataset = (
        read_dataset_from_netcdf(Path(__file__).parent / "regression_results" / f"{case_name}_result.nc")
        .load()
        .drop_vars(ignored_variables, errors="ignore")
    )

    ordered_dims = [dim for dim in reference_dataset.dims]
    assert_allclose(dataset.transpose(*ordered_dims), reference_dataset.transpose(*ordered_dims))


@pytest.mark.regression
@pytest.mark.parametrize("case", ALL_CASE_PATHS, ids=ALL_CASE_NAMES)
@pytest.mark.filterwarnings("ignore:Not all input parameters were used")
def test_regression_against_case_with_update(case: Path):
    input_parameters, algorithm, _, _ = read_case(case)
    case_name = case.parent.stem

    dataset = xr.Dataset(input_parameters)

    dataset = algorithm.update_dataset(dataset)
    write_dataset_to_netcdf(dataset, Path(__file__).parent / "regression_results" / f"test2_{case.parent.stem}.nc")

    dataset = (
        read_dataset_from_netcdf(Path(__file__).parent / "regression_results" / f"test2_{case.parent.stem}.nc")
        .load()
        .drop_vars(ignored_variables, errors="ignore")
    )
    reference_dataset = (
        read_dataset_from_netcdf(Path(__file__).parent / "regression_results" / f"{case_name}_result.nc")
        .load()
        .drop_vars(ignored_variables, errors="ignore")
    )

    ordered_dims = [dim for dim in reference_dataset.dims]
    assert_allclose(dataset.transpose(*ordered_dims), reference_dataset.transpose(*ordered_dims))


@pytest.mark.regression
@pytest.mark.filterwarnings("ignore:Not all input parameters were used")
def test_regression_against_case_with_repeated_update():
    case = CASES_DIR / "SPARC_PRD" / "input.yaml"

    input_parameters, algorithm, _, _ = read_case(case)
    input_parameters["average_electron_density"] = input_parameters["average_electron_density"][::5]
    input_parameters["average_electron_temp"] = input_parameters["average_electron_temp"][::5]

    dataset = xr.Dataset(input_parameters)

    first_run = algorithm.update_dataset(dataset)

    # Make sure that first_run isn't being modified in-place when we re-run it.
    first_run_copy = copy.deepcopy(first_run)
    second_run = algorithm.update_dataset(first_run_copy)

    for variable in ["atomic_data"]:
        first_run = first_run.drop_vars(variable)
        second_run = second_run.drop_vars(variable)

    # The ordering of the dimensions changes between the runs, and for some reason the automatic
    # xarray broadcasting isn't handling this. Because of this, we manually ensure that the
    # dimension ordering matches.
    ordered_dims = [dim for dim in first_run.dims]
    assert_allclose(
        first_run.transpose(*ordered_dims),
        second_run.transpose(*ordered_dims),
    )
