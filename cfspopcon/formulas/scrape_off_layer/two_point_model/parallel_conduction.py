"""Calculate quantities used in alternative parallel conduction models."""

from typing import Union

import xarray as xr
import numpy as np

from ....unit_handling import Quantity, Unitfull, ureg, wraps_ufunc


def calc_Spitzer_conduction_reduction_factor_scaling(
        upstream_SOL_collisionality: Union[Quantity, xr.DataArray],
) -> Union[float, xr.DataArray]:
    
    Spitzer_conduction_reduction_factor = 0.696*np.exp(-8.059*(upstream_SOL_collisionality)**-1.074)+0.260
    Spitzer_conduction_reduction_factor = xr.where(Spitzer_conduction_reduction_factor < 0.5, 1./(1+6.584/upstream_SOL_collisionality),Spitzer_conduction_reduction_factor)                 

    
    return Spitzer_conduction_reduction_factor


@wraps_ufunc(
    return_units=dict(Spitzer_conduction_reduction_factor=ureg.dimensionless),
    input_units=dict(
        separatrix_electron_density=ureg.m**-3,
        separatrix_electron_temp=ureg.eV,
        parallel_connection_length=ureg.m,
        target_electron_temp=ureg.eV,
        kappa_e0=ureg.W / (ureg.eV**3.5 * ureg.m),
        electron_mass=ureg.kg,
        electron_charge=ureg.C,
        SOL_conduction_fraction=ureg.dimensionless,
        flux_limit_factor_alpha=ureg.dimensionless,
    ),
)
def calc_Spitzer_conduction_reduction_factor_fluxlim(
    separatrix_electron_density: Union[Quantity, xr.DataArray],
    separatrix_electron_temp: Union[Quantity, xr.DataArray],
    parallel_connection_length: Union[Quantity, xr.DataArray],
    target_electron_temp: Union[Quantity, xr.DataArray],
    kappa_e0: Union[Quantity, xr.DataArray],
    electron_mass: Unitfull = 9.1e-31 * ureg.kg,
    electron_charge: Unitfull = 1.602e-19 * ureg.C,
    SOL_conduction_fraction: Union[float, xr.DataArray] = 1.0,
    flux_limit_factor_alpha: Union[float, xr.DataArray] = 0.15,
) -> Union[float, xr.DataArray]:

    spitzer_heat_flux = (
        (2.0 / 7) * (kappa_e0 / parallel_connection_length) * ((separatrix_electron_temp) ** 3.5 - (target_electron_temp) ** 3.5)
    )
    free_streaming_heat_flux = separatrix_electron_density * (separatrix_electron_temp * electron_charge) ** 1.5 * (1.0 / electron_mass) ** 0.5
    Spitzer_conduction_reduction_factor = 1 / (1 + spitzer_heat_flux / (flux_limit_factor_alpha * free_streaming_heat_flux))

    return Spitzer_conduction_reduction_factor


def calc_delta_electron_sheath_factor(
        separatrix_electron_temp: Union[Quantity, xr.DataArray],
        target_electron_temp: Union[Quantity, xr.DataArray], 
        SOL_momentum_loss_fraction: Union[float, xr.DataArray] = 0.0,
) -> Union[float, xr.DataArray]:

    delta_electron_sheath_factor = 1.08 * (1-SOL_momentum_loss_fraction) * (separatrix_electron_temp/target_electron_temp)**0.25
    
    return delta_electron_sheath_factor
    
