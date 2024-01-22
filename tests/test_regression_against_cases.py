from pathlib import Path

import pytest
import xarray as xr
from regression_results.generate_regression_results import (
    ALL_CASE_NAMES,
    ALL_CASE_PATHS,
)
from xarray.testing import assert_allclose

from cfspopcon.file_io import write_dataset_to_netcdf, read_dataset_from_netcdf
from cfspopcon.input_file_handling import read_case


@pytest.mark.parametrize("case", ALL_CASE_PATHS, ids=ALL_CASE_NAMES)
@pytest.mark.filterwarnings("ignore:Not all input parameters were used")
def test_regression_against_case(case: Path):
    input_parameters, algorithm, _ = read_case(case)
    case_name = case.parent.stem

    dataset = algorithm.run(**input_parameters)
    write_dataset_to_netcdf(dataset, Path(__file__).parent / "regression_results" / f"test1_{case.parent.stem}.nc")

    reference_dataset = read_dataset_from_netcdf(Path(__file__).parent / "regression_results" / f"{case_name}_result.nc").load()

    assert_allclose(dataset, reference_dataset, rtol=1e-8, atol=0)


@pytest.mark.parametrize("case", ALL_CASE_PATHS, ids=ALL_CASE_NAMES)
@pytest.mark.filterwarnings("ignore:Not all input parameters were used")
def test_regression_against_case_with_update(case: Path):
    input_parameters, algorithm, _ = read_case(case)
    case_name = case.parent.stem

    dataset = xr.Dataset(input_parameters)

    for alg in algorithm.algorithms:  # type: ignore
        dataset = alg.update_dataset(dataset)

    write_dataset_to_netcdf(dataset, Path(__file__).parent / "regression_results" / f"test2_{case.parent.stem}.nc")

    reference_dataset = read_dataset_from_netcdf(Path(__file__).parent / "regression_results" / f"{case_name}_result.nc").load()

    assert_allclose(dataset, reference_dataset, rtol=1e-8, atol=0)
