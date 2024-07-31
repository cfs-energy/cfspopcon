"""Compute the total radiated power."""

import numpy as np
import xarray as xr

from ....named_options import RadiationMethod
from ....unit_handling import Unitfull, ureg
from ...atomic_data import AtomicData
from .mavrin_coronal import calc_impurity_radiated_power_mavrin_coronal
from .mavrin_noncoronal import calc_impurity_radiated_power_mavrin_noncoronal
from .post_and_jensen import calc_impurity_radiated_power_post_and_jensen
from .radas import calc_impurity_radiated_power_radas


def calc_impurity_radiated_power(
    radiated_power_method: RadiationMethod,
    rho: Unitfull,
    electron_temp_profile: Unitfull,
    electron_density_profile: Unitfull,
    impurity_concentration: xr.DataArray,
    plasma_volume: Unitfull,
    atomic_data: AtomicData,
) -> xr.DataArray:
    """Compute the total radiated power due to fuel and impurity species.

    Args:
        radiated_power_method: [] :term:`glossary link<radiated_power_method>`
        rho: [~] :term:`glossary link<rho>`
        electron_temp_profile: [keV] :term:`glossary link<electron_temp_profile>`
        electron_density_profile: [1e19 m^-3] :term:`glossary link<electron_density_profile>`
        impurity_concentration: [] :term:`glossary link<impurity_concentration>`
        plasma_volume: [m^3] :term:`glossary link<plasma_volume>`
        atomic_data: :term:`glossary link<atomic_data>`

    Returns:
         [MW] Estimated radiation power due to this impurity
    """
    P_rad_kwargs = dict(
        rho=rho,
        electron_temp_profile=electron_temp_profile,
        electron_density_profile=electron_density_profile,
        impurity_concentration=impurity_concentration,
        impurity_species=impurity_concentration.dim_species,
        plasma_volume=plasma_volume,
    )
    if radiated_power_method == RadiationMethod.PostJensen:
        P_rad_impurity = calc_impurity_radiated_power_post_and_jensen(**P_rad_kwargs)
    elif radiated_power_method == RadiationMethod.MavrinCoronal:
        P_rad_impurity = calc_impurity_radiated_power_mavrin_coronal(**P_rad_kwargs)
    elif radiated_power_method == RadiationMethod.MavrinNoncoronal:
        P_rad_impurity = calc_impurity_radiated_power_mavrin_noncoronal(**P_rad_kwargs, tau_i=np.inf * ureg.s)
    elif radiated_power_method == RadiationMethod.Radas:
        P_rad_impurity = calc_impurity_radiated_power_radas(**P_rad_kwargs, atomic_data=atomic_data)
    else:
        raise NotImplementedError(f"No implementation for radiated_power_method = {radiated_power_method}")

    return P_rad_impurity  # type:ignore[no-any-return]
