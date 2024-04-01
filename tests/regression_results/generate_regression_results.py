from pathlib import Path

import xarray as xr

from cfspopcon.file_io import write_dataset_to_netcdf, write_point_to_file
from cfspopcon.input_file_handling import read_case

CASES_DIR = Path(__file__).parent.parent.parent / "example_cases"
ALL_CASE_PATHS = list(CASES_DIR.rglob("input.yaml"))
ALL_CASE_NAMES = [path.parent.relative_to(CASES_DIR).stem for path in ALL_CASE_PATHS]

if __name__ == "__main__":
    for case in ALL_CASE_PATHS:
        input_parameters, algorithm, points, _ = read_case(case)

        dataset = xr.Dataset(input_parameters)

        dataset = algorithm.update_dataset(dataset)

        write_dataset_to_netcdf(dataset, Path(__file__).parent / f"{case.parent.stem}_result.nc")

        for point, point_params in points.items():
            write_point_to_file(dataset, point, point_params, output_dir=Path(__file__).parent)
