"""Calculate the concentration and effect of a core radiator required to achieve above a defined core radiative fraction."""

import numpy as np
import xarray as xr

from ... import named_options
from ...algorithm_class import Algorithm
from ...helpers import make_impurities_array
from ...unit_handling import Unitfull
from .. import radiated_power, zeff_and_dilution


@Algorithm.register_algorithm(
    return_keys=[
        "core_radiator_concentration",
        "P_radiated_by_core_radiator",
        "P_radiation",
        "core_radiator_concentration",
        "core_radiator_charge_state",
        "zeff_change_from_core_rad",
        "dilution_change_from_core_rad",
        "z_effective",
        "dilution",
    ]
)
def calc_extrinsic_core_radiator(
    minimum_core_radiated_fraction: Unitfull,
    P_in: Unitfull,
    P_radiation: Unitfull,
    average_electron_density: Unitfull,
    average_electron_temp: Unitfull,
    z_effective: Unitfull,
    dilution: Unitfull,
    rho: Unitfull,
    electron_density_profile: Unitfull,
    electron_temp_profile: Unitfull,
    plasma_volume: Unitfull,
    radiated_power_method: named_options.RadiationMethod,
    radiated_power_scalar: Unitfull,
    core_radiator: named_options.AtomicSpecies,
    atomic_data: xr.DataArray,
) -> tuple[Unitfull, ...]:
    """Calculate the concentration and effect of a core radiator required to achieve above a defined core radiative fraction.

    Args:
        minimum_core_radiated_fraction: :term:`glossary link<minimum_core_radiated_fraction>`
        P_in: :term:`glossary link<P_in>`
        P_radiation: :term:`glossary link<P_radiation>`
        average_electron_density: :term:`glossary link<average_electron_density>`
        average_electron_temp: :term:`glossary link<average_electron_temp>`
        z_effective: :term:`glossary link<z_effective>`
        dilution: :term:`dilution`
        rho: :term:`glossary link<rho>`
        electron_density_profile: :term:`glossary link<electron_density_profile>`
        electron_temp_profile: :term:`glossary link<electron_temp_profile>`
        plasma_volume: :term:`glossary link<plasma_volume>`
        radiated_power_method: :term:`glossary link<radiated_power_method>`
        radiated_power_scalar: :term:`radiated_power_scalar`
        core_radiator: :term:`glossary link<core_radiator>`
        atomic_data: :term:`glossary link<atomic_data>`

    Returns:
        :term:`core_radiator_concentration`, :term:`P_radiated_by_core_radiator`, :term:`P_radiation`, :term:`core_radiator_concentration`, :term:`core_radiator_charge_state`, :term:`zeff_change_from_core_rad` :term:`dilution_change_from_core_rad`, :term:`z_effective`, :term:`dilution`

    """
    # Force P_radiated_by_core_radiator to be >= 0.0 (core radiator cannot reduce radiated power)
    P_radiated_by_core_radiator = np.maximum(
        minimum_core_radiated_fraction * P_in - P_radiation, 0.0
    )
    P_radiation = np.maximum(minimum_core_radiated_fraction * P_in, P_radiation)

    P_rad_per_core_radiator = (
        radiated_power_scalar
        * radiated_power.impurity_radiated_power.calc_impurity_radiated_power(
            radiated_power_method=(
                named_options.RadiationMethod.Radas
                if radiated_power_method == named_options.RadiationMethod.Inherent
                else radiated_power_method
            ),
            rho=rho,
            electron_temp_profile=electron_temp_profile,
            electron_density_profile=electron_density_profile,
            impurities=make_impurities_array(core_radiator, 1.0),
            plasma_volume=plasma_volume,
            atomic_data=atomic_data.item(),
        ).sum(dim="dim_species")
    )
    core_radiator_concentration = xr.where(  # type:ignore[no-untyped-call]
        P_radiated_by_core_radiator > 0,
        P_radiated_by_core_radiator / P_rad_per_core_radiator,
        0.0,
    )

    core_radiator_charge_state = (
        zeff_and_dilution.impurity_charge_state.calc_impurity_charge_state(
            average_electron_density,
            average_electron_temp,
            core_radiator,
            atomic_data.item(),
        )
    )
    zeff_change_from_core_rad = (
        zeff_and_dilution.zeff_and_dilution_from_impurities.calc_change_in_zeff(
            core_radiator_charge_state, core_radiator_concentration
        )
    )
    dilution_change_from_core_rad = (
        zeff_and_dilution.zeff_and_dilution_from_impurities.calc_change_in_dilution(
            core_radiator_charge_state, core_radiator_concentration
        )
    )

    z_effective = z_effective + zeff_change_from_core_rad
    dilution = (dilution - dilution_change_from_core_rad).clip(min=0.0)

    return (
        core_radiator_concentration,
        P_radiated_by_core_radiator,
        P_radiation,
        core_radiator_concentration,
        core_radiator_charge_state,
        zeff_change_from_core_rad,
        dilution_change_from_core_rad,
        z_effective,
        dilution,
    )
