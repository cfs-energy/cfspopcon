"""Calculate the concentration and effect of a core radiator required to achieve above a defined core radiative fraction."""

import numpy as np
import xarray as xr

from ... import named_options
from ...algorithm_class import Algorithm
from ...helpers import extend_impurities_array, make_impurities_array
from ...unit_handling import Unitfull
from .. import radiated_power


@Algorithm.register_algorithm(return_keys=["min_P_radiation"])
def calc_min_P_radiation_from_fraction(minimum_core_radiated_fraction: Unitfull, P_in: Unitfull) -> Unitfull:
    """Set the minimum radiated power as a fraction of the total input power.

    Args:
        minimum_core_radiated_fraction: :term:`glossary link<minimum_core_radiated_fraction>`
        P_in: :term:`glossary link<P_in>`

    Returns:
        :term:`min_P_radiation`
    """
    return minimum_core_radiated_fraction * P_in


@Algorithm.register_algorithm(return_keys=["min_P_radiation"])
def calc_min_P_radiation_from_LH_factor(maximum_P_LH_factor_for_P_SOL: Unitfull, P_LH_thresh: Unitfull, P_in: Unitfull) -> Unitfull:
    """Set the minimum radiated power such that P_sol doesn't go above some multiple of the LH threshold power.

    Args:
        maximum_P_LH_factor_for_P_SOL: :term:`glossary link<maximum_P_LH_factor_for_P_SOL>`
        P_LH_thresh: :term:`glossary link<P_LH_thresh>`
        P_in: :term:`glossary link<P_in>`

    Returns:
        :term:`min_P_radiation`
    """
    return np.maximum(P_in - maximum_P_LH_factor_for_P_SOL * P_LH_thresh, 0.0)


@Algorithm.register_algorithm(return_keys=["P_radiation", "P_radiation_from_core_seeded_impurity"])
def calc_P_radiation_from_core_seeded_impurity(P_radiation: Unitfull, min_P_radiation: Unitfull) -> tuple[Unitfull, ...]:
    """Increases P_radiation until it is at least min_P_radiation.

    If P_radiation > min_P_radiation, this does nothing.

    If P_radiation < min_P_radiation, it is assumed some core seed impurity will be
    injected to increased P_radiation until it reaches min_P_radiation. The additional
    radiated power is returned as P_radiation_from_core_seeded_impurity

    Args:
        P_radiation: :term:`glossary link<P_radiation>`
        min_P_radiation: :term:`glossary link<min_P_radiation>`

    Returns:
        :term:`P_radiation`, :term:`P_radiation_from_core_seeded_impurity`
    """
    # Force P_radiated_by_core_impurity_species to be >= 0.0 (core radiator cannot reduce radiated power)
    P_radiation_from_core_seeded_impurity = np.maximum(min_P_radiation - P_radiation, 0.0)

    # Compute the new value of P_radiation after accounting for seeding
    P_radiation = np.maximum(min_P_radiation, P_radiation)

    return P_radiation, P_radiation_from_core_seeded_impurity


@Algorithm.register_algorithm(
    return_keys=[
        "impurities",
    ]
)
def calc_core_seeded_impurity_concentration(
    P_radiation_from_core_seeded_impurity: Unitfull,
    impurities: Unitfull,
    rho: Unitfull,
    electron_density_profile: Unitfull,
    electron_temp_profile: Unitfull,
    plasma_volume: Unitfull,
    radiated_power_method: named_options.RadiationMethod,
    radiated_power_scalar: Unitfull,
    core_impurity_species: named_options.AtomicSpecies,
    atomic_data: xr.DataArray,
) -> tuple[Unitfull, ...]:
    """Calculate the concentration of a core radiator required to increase the radiated power by a desired amount.

    Args:
        P_radiation_from_core_seeded_impurity: :term:`glossary link<P_radiation_from_core_seeded_impurity>`
        impurities: :term:`glossary link<impurities>`
        rho: :term:`glossary link<rho>`
        electron_density_profile: :term:`glossary link<electron_density_profile>`
        electron_temp_profile: :term:`glossary link<electron_temp_profile>`
        plasma_volume: :term:`glossary link<plasma_volume>`
        radiated_power_method: :term:`glossary link<radiated_power_method>`
        radiated_power_scalar: :term:`glossary link<radiated_power_scalar>`
        core_impurity_species: :term:`glossary link<core_impurity_species>`
        atomic_data: :term:`glossary link<atomic_data>`

    Returns:
        :term:`impurities`
    """
    if radiated_power_method == named_options.RadiationMethod.Inherent:
        radiated_power_method_for_core_impurity = named_options.RadiationMethod.Radas
    else:
        radiated_power_method_for_core_impurity = radiated_power_method

    kwargs = dict(
        radiated_power_method=radiated_power_method_for_core_impurity,
        rho=rho,
        electron_temp_profile=electron_temp_profile,
        electron_density_profile=electron_density_profile,
        plasma_volume=plasma_volume,
        atomic_data=atomic_data.item(),
    )

    P_radiated_per_unit_concentration = radiated_power_scalar * radiated_power.impurity_radiated_power.calc_impurity_radiated_power(
        **kwargs,
        impurities=make_impurities_array(core_impurity_species, 1.0),
    ).sum(dim="dim_species")

    core_seeded_impurity_concentration = xr.where(  # type:ignore[no-untyped-call]
        P_radiation_from_core_seeded_impurity > 0, P_radiation_from_core_seeded_impurity / P_radiated_per_unit_concentration, 0.0
    )

    impurities = extend_impurities_array(impurities, core_impurity_species, core_seeded_impurity_concentration)

    return impurities
