"""Open the atomic data files and return corresponding xr.Datasets and interpolators."""
from pathlib import Path
from typing import Optional, Union

import numpy as np
import xarray as xr
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator  # type: ignore[import]

from ..named_options import Impurity
from ..unit_handling import convert_units, magnitude, ureg


def read_atomic_data(directory: Optional[Path] = None) -> dict[Impurity, xr.DataArray]:
    """Read the atomic data files and return a dictionary mapping Impurity keys to xr.DataArrays of atomic data."""
    if directory is None:
        directory = Path(__file__).parent

    atomic_data_files = {
        Impurity.Helium: directory / "helium.nc",
        Impurity.Lithium: directory / "lithium.nc",
        Impurity.Beryllium: directory / "beryllium.nc",
        Impurity.Carbon: directory / "carbon.nc",
        Impurity.Nitrogen: directory / "nitrogen.nc",
        Impurity.Oxygen: directory / "oxygen.nc",
        Impurity.Neon: directory / "neon.nc",
        Impurity.Argon: directory / "argon.nc",
        Impurity.Krypton: directory / "krypton.nc",
        Impurity.Xenon: directory / "xenon.nc",
        Impurity.Tungsten: directory / "tungsten.nc",
    }

    atomic_data = {}

    for key, file in atomic_data_files.items():
        if not file.exists():
            raise FileNotFoundError(f"Could not find the atomic data file {file.absolute()} for species {key}")

        ds = xr.open_dataset(file).pint.quantify()

        # Convert the dimensions from linear to log values
        ds["dim_log_electron_temperature"] = xr.DataArray(
            np.log10(magnitude(convert_units(ds.electron_temperature, ureg.eV))), dims="dim_electron_temperature"
        )
        ds["dim_log_electron_density"] = xr.DataArray(
            np.log10(magnitude(convert_units(ds.electron_density, ureg.m**-3))), dims="dim_electron_density"
        )
        ds["dim_log_ne_tau"] = xr.DataArray(np.log10(magnitude(convert_units(ds.ne_tau, ureg.m**-3 * ureg.s))), dims="dim_ne_tau")

        ds = ds.swap_dims({"dim_electron_temperature": "dim_log_electron_temperature"})
        ds = ds.swap_dims({"dim_electron_density": "dim_log_electron_density"})
        ds = ds.swap_dims({"dim_ne_tau": "dim_log_ne_tau"})

        def build_interpolator(curve: xr.Dataset) -> Union[RectBivariateSpline, RegularGridInterpolator]:
            if curve.ndim == 2:
                return RectBivariateSpline(
                    x=curve.dim_log_electron_temperature,
                    y=curve.dim_log_electron_density,
                    z=np.log10(magnitude(curve.transpose("dim_log_electron_temperature", "dim_log_electron_density"))),  # type: ignore[arg-type]
                )
            elif curve.ndim == 3:
                return RegularGridInterpolator(
                    points=(curve.dim_log_electron_temperature, curve.dim_log_electron_density, curve.dim_log_ne_tau),
                    values=np.log10(magnitude(curve.transpose("dim_log_electron_temperature", "dim_log_electron_density", "dim_log_ne_tau"))),  # type: ignore[arg-type]
                    bounds_error=False,
                    method="linear",
                )
            else:
                raise NotImplementedError(f"Cannot build an interpolator for a curve with ndim={curve.ndim}")

        ds = ds.assign_attrs(
            coronal_Lz_interpolator=build_interpolator(ds.coronal_electron_emission_prefactor),
            coronal_mean_Z_interpolator=build_interpolator(ds.coronal_mean_charge_state),
            noncoronal_Lz_interpolator=build_interpolator(ds.noncoronal_electron_emission_prefactor),
            noncoronal_mean_Z_interpolator=build_interpolator(ds.noncoronal_mean_charge_state),
        )

        # Drop the linear coordinates
        atomic_data[key] = ds.drop_vars(("dim_ne_tau", "dim_electron_temperature", "dim_electron_density"))

    return atomic_data
