"""Class definition for CoeffInterpolator which is used to interpolate atomic data."""

from functools import partial
from typing import Union

import numpy as np
import xarray as xr
from scipy.interpolate import RectBivariateSpline

from cfspopcon.unit_handling import Quantity, get_units, magnitude_in_units, ureg, wraps_ufunc


class CoeffInterpolator(RectBivariateSpline):
    """An extension of a 2D spline interpolator which handles interpolations in log-space, for interpolating atomic data."""

    tiny = np.finfo(np.float64).tiny

    def log10_with_floor(self, x: xr.DataArray | np.ndarray | float) -> xr.DataArray | np.ndarray | float:
        """Return the log of x if x > 0, and otherwise return the log of the smallest representable float."""
        return np.log10(np.maximum(x, self.tiny))  # type: ignore[no-any-return]

    def __init__(
        self,
        coeff: xr.DataArray,
        reference_electron_density: Quantity = 1.0 * ureg.m**-3,
        reference_electron_temp: Quantity = 1.0 * ureg.eV,
    ) -> None:
        """Builds a bivariate spline interpolator for the provided DataArray of coefficients.

        The interpolator performs interpolations on the log of the provided coefficient, for the log of the
        provided temperature and density, since the atomic data arrays are provided with log-spaced values.

        Parameters:
        - coeff (xr.DataArray): The xarray DataArray containing the data to interpolate, indexed by electron temperature and density.
        - reference_electron_density (Quantity): the normalization value for the electron density coordinates.
        - reference_electron_temp (Quantity): the normalization value for the electron temp coordinates.

        Returns:
        - CoeffInterpolator: The bivariate spline interpolator object with eval, vector_eval and grid_eval methods.
        """
        assert np.all(coeff.dim_electron_density > 0.0), "Encountered null or negative values in electron density for CoeffInterpolator"

        assert np.all(coeff.dim_electron_temp > 0.0), "Encountered null or negative values in electron temp for CoeffInterpolator"

        assert np.all(coeff >= 0.0), "Encountered negative values in coeff for CoeffInterpolator"

        self.units = get_units(coeff)
        coeff_magnitude = magnitude_in_units(coeff.transpose("dim_electron_temp", "dim_electron_density"), self.units)

        self.electron_density_units = get_units(reference_electron_density)
        self.electron_temp_units = get_units(reference_electron_temp)
        # Get the normalization of the electron density and electron temp. Usually the reference values are 1.0 * units.
        # This prevents an error if there is a scalar prefactor i.e. 2.0 * units
        electron_density_norm = magnitude_in_units(reference_electron_density, self.electron_density_units)
        electron_temp_norm = magnitude_in_units(reference_electron_temp, self.electron_temp_units)

        electron_temp_coord = coeff.dim_electron_temp / electron_temp_norm
        electron_density_coord = coeff.dim_electron_density / electron_density_norm

        self.coeff_is_zero = np.all(coeff == 0.0)

        if self.coeff_is_zero:
            super().__init__(
                self.log10_with_floor(electron_temp_coord),  # type: ignore[arg-type]
                self.log10_with_floor(electron_density_coord),  # type: ignore[arg-type]
                np.zeros_like(coeff_magnitude),
            )
        else:
            assert np.all(coeff > 0.0), "Encountered mix of null and positive values in coeff for CoeffInterpolator"

            super().__init__(
                self.log10_with_floor(electron_temp_coord),  # type: ignore[arg-type]
                self.log10_with_floor(electron_density_coord),  # type: ignore[arg-type]
                self.log10_with_floor(coeff_magnitude),  # type: ignore[arg-type]
            )

        self.electron_temp_coord = electron_temp_coord
        self.electron_density_coord = electron_density_coord

        self.max_temp = np.max(electron_temp_coord).item()
        self.min_temp = np.min(electron_temp_coord).item()
        self.max_density = np.max(electron_density_coord).item()
        self.min_density = np.min(electron_density_coord).item()

        call_config = dict(
            input_units=dict(
                electron_density=self.electron_density_units,
                electron_temp=self.electron_temp_units,
                allow_extrap=None,
                grid_from_inputs=None,
            ),
            return_units=dict(coeff=self.units),
            pass_as_kwargs=("allow_extrap", "grid_from_inputs"),
        )

        self.eval = wraps_ufunc(**call_config)(self.__call__)  # type: ignore[arg-type]
        self.vector_eval = wraps_ufunc(**call_config)(partial(self.__call__, grid_from_inputs=False))  # type: ignore[arg-type]
        self.grid_eval = wraps_ufunc(
            **call_config,  # type: ignore[arg-type]
            input_core_dims=(("dim_electron_density",), ("dim_electron_temp",)),
            output_core_dims=(("dim_electron_temp", "dim_electron_density"),),
        )(partial(self.__call__, grid_from_inputs=True))

    def __call__(  # type: ignore[override]
        self,
        electron_density: float | np.ndarray,
        electron_temp: float | np.ndarray,
        allow_extrap: bool = False,
        grid_from_inputs: bool = False,
    ) -> np.ndarray:
        """Interpolate the atomic data to the requested point, handling the log-spaced interpolation but not handling units.

        For unit-aware interpolation, use the eval, vector_eval or grid_eval methods which are dynamically constructed in the __init__ function.

        NOTE: Unlike RectBivariateSpline, the "grid_from_inputs" option (mapping to 'grid' in RectBivariateSpline) is false by default.
        """
        if allow_extrap:
            electron_temp = np.minimum(electron_temp, self.max_temp)
            electron_temp = np.maximum(electron_temp, self.min_temp)
            electron_density = np.minimum(electron_density, self.max_density)
            electron_density = np.maximum(electron_density, self.min_density)
        else:
            assert (
                (np.min(electron_density) >= self.min_density)
                & (np.max(electron_density) <= self.max_density)
                & (np.min(electron_temp) >= self.min_temp)
                & (np.max(electron_temp) <= self.max_temp)
            ), "CoeffInterpolator error: values off grid and allow_extrap=False. "

        values = super().__call__(np.log10(electron_temp), np.log10(electron_density), grid=grid_from_inputs)

        if self.coeff_is_zero:
            return values
        else:
            return np.power(10, values)
