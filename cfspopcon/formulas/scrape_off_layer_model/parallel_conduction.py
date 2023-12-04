"""Calculate quantities used in alternative parallel conduction models."""

from typing import Union

import xarray as xr

from ...unit_handling import Quantity, Unitfull, ureg, wraps_ufunc


@wraps_ufunc(
    return_units=dict(Spitzer_conduction_reduction_factor=ureg.dimensionless),
    input_units=dict(
        upstream_electron_density=ureg.m**-3,
        upstream_electron_temp=ureg.eV,
        parallel_connection_length=ureg.m,
        target_electron_temp=ureg.eV,
        kappa_e0=ureg.W/(ureg.eV**3.5 * ureg.m),
        electron_mass=ureg.kg,
        electron_charge=ureg.C,
        SOL_conduction_fraction=ureg.dimensionless,
        flux_limit_factor_alpha=ureg.dimensionless,
    ),
)
def calc_Spitzer_conduction_reduction_factor_fluxlim(
        upstream_electron_density: Union[Quantity, xr.DataArray],
        upstream_electron_temp: Union[Quantity, xr.DataArray],
        parallel_connection_length: Union[Quantity, xr.DataArray],
        target_electron_temp: Union[Quantity, xr.DataArray],
        kappa_e0: Union[Quantity, xr.DataArray],
        electron_mass: Unitfull = 9.1e-31 * ureg.kg,
        electron_charge: Unitfull = 1.602e-19 * ureg.C,
        SOL_conduction_fraction: Union[float, xr.DataArray] = 1.0,
        flux_limit_factor_alpha: Union[float, xr.DataArray] = 0.15,
    ) -> Union[float, xr.DataArray]:

    spitzer_heat_flux = ((2./7) * (kappa_e0/parallel_connection_length) * ((upstream_electron_temp)**3.5 - (target_electron_temp)**3.5))
    free_streaming_heat_flux = upstream_electron_density*(upstream_electron_temp * electron_charge)**1.5 * (1./electron_mass)**0.5
    Spitzer_conduction_reduction_factor = 1/(1+spitzer_heat_flux/(flux_limit_factor_alpha*free_streaming_heat_flux))

    return Spitzer_conduction_reduction_factor

