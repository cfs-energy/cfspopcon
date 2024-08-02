"""Calculate the power radiated from the confined region due to the fuel and impurity species."""

import xarray as xr

from ... import named_options
from ...algorithm_class import Algorithm
from ...unit_handling import Unitfull
from .bremsstrahlung import calc_bremsstrahlung_radiation
from .impurity_radiated_power import calc_impurity_radiated_power
from .synchrotron import calc_synchrotron_radiation


@Algorithm.register_algorithm(return_keys=["P_radiation"])
def calc_intrinsic_radiated_power_from_core(
    rho: Unitfull,
    electron_density_profile: Unitfull,
    electron_temp_profile: Unitfull,
    z_effective: Unitfull,
    plasma_volume: Unitfull,
    major_radius: Unitfull,
    minor_radius: Unitfull,
    magnetic_field_on_axis: Unitfull,
    separatrix_elongation: Unitfull,
    radiated_power_method: named_options.RadiationMethod,
    radiated_power_scalar: Unitfull,
    impurity_concentration: xr.DataArray,
    atomic_data: xr.DataArray,
) -> Unitfull:
    """Calculate the power radiated from the confined region due to the fuel and impurity species.

    Args:
        rho: :term:`glossary link<rho>`
        electron_density_profile: :term:`glossary link<electron_density_profile>`
        electron_temp_profile: :term:`glossary link<electron_temp_profile>`
        z_effective: :term:`glossary link<z_effective>`
        plasma_volume: :term:`glossary link<plasma_volume>`
        major_radius: :term:`glossary link<major_radius>`
        minor_radius: :term:`glossary link<minor_radius>`
        magnetic_field_on_axis: :term:`glossary link<magnetic_field_on_axis>`
        separatrix_elongation: :term:`glossary link<separatrix_elongation>`
        radiated_power_method: :term:`glossary link<radiated_power_method>`
        radiated_power_scalar: :term:`glossary link<radiated_power_scalar>`
        impurity_concentration: :term:`glossary link<impurity_concentration>`
        atomic_data: :term:`glossary link<atomic_data>`

    Returns:
        :term:`P_radiation`

    """
    P_rad_bremsstrahlung = calc_bremsstrahlung_radiation(rho, electron_density_profile, electron_temp_profile, z_effective, plasma_volume)
    P_rad_bremsstrahlung_from_hydrogen = calc_bremsstrahlung_radiation(
        rho, electron_density_profile, electron_temp_profile, 1.0, plasma_volume
    )
    P_rad_synchrotron = calc_synchrotron_radiation(
        rho,
        electron_density_profile,
        electron_temp_profile,
        major_radius,
        minor_radius,
        magnetic_field_on_axis,
        separatrix_elongation,
        plasma_volume,
    )

    # Calculate radiated power due to Bremsstrahlung, Synchrotron and impurities
    if radiated_power_method == named_options.RadiationMethod.Inherent:
        return radiated_power_scalar * (P_rad_bremsstrahlung + P_rad_synchrotron)
    else:
        P_rad_impurity = calc_impurity_radiated_power(
            radiated_power_method=radiated_power_method,
            rho=rho,
            electron_temp_profile=electron_temp_profile,
            electron_density_profile=electron_density_profile,
            impurity_concentration=impurity_concentration,
            plasma_volume=plasma_volume,
            atomic_data=atomic_data.item(),
        )

        return radiated_power_scalar * (P_rad_bremsstrahlung_from_hydrogen + P_rad_synchrotron + P_rad_impurity.sum(dim="dim_species"))
