"""Calculate the power radiated from the confined region due to the fuel and impurity species."""
import xarray as xr

from .. import formulas, named_options
from ..atomic_data import read_atomic_data
from ..unit_handling import Unitfull, convert_to_default_units
from .algorithm_class import Algorithm

RETURN_KEYS = ["P_radiation"]


def run_calc_core_radiated_power(
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
    impurities: xr.DataArray,
) -> dict[str, Unitfull]:
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
        impurities: :term:`glossary link<impurities>`

    Returns:
        :term:`P_radiation`

    """
    P_rad_bremsstrahlung = formulas.calc_bremsstrahlung_radiation(
        rho, electron_density_profile, electron_temp_profile, z_effective, plasma_volume
    )
    P_rad_bremsstrahlung_from_hydrogen = formulas.calc_bremsstrahlung_radiation(
        rho, electron_density_profile, electron_temp_profile, 1.0, plasma_volume
    )
    P_rad_synchrotron = formulas.calc_synchrotron_radiation(
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
        P_radiation = radiated_power_scalar * (P_rad_bremsstrahlung + P_rad_synchrotron)
    else:
        atomic_data = read_atomic_data()

        P_rad_impurity = formulas.calc_impurity_radiated_power(
            radiated_power_method=radiated_power_method,
            rho=rho,
            electron_temp_profile=electron_temp_profile,
            electron_density_profile=electron_density_profile,
            impurities=impurities,
            plasma_volume=plasma_volume,
            atomic_data=atomic_data,
        )

        P_radiation = radiated_power_scalar * (
            P_rad_bremsstrahlung_from_hydrogen + P_rad_synchrotron + P_rad_impurity.sum(dim="dim_species")
        )

    local_vars = locals()
    return {key: convert_to_default_units(local_vars[key], key) for key in RETURN_KEYS}


calc_core_radiated_power = Algorithm(
    function=run_calc_core_radiated_power,
    return_keys=RETURN_KEYS,
)
