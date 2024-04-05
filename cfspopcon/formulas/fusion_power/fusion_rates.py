"""Calculate fusion power and corresponding neutron wall loading."""

from typing import Union

from numpy import float64
from numpy.typing import NDArray

from ...algorithm_class import Algorithm
from ...named_options import ReactionType
from ...unit_handling import ureg, wraps_ufunc
from ..geometry.volume_integral import integrate_profile_over_volume
from .reaction_energies import ENERGY
from .reaction_rate_coefficients import SIGMAV
from .fusion_data import DTFusionBoschHale, DTFusionHively
import xarray as xr


@Algorithm.register_algorithm(return_keys=["P_fusion", "P_neutron", "P_alpha"])
def calc_fusion_power(
    fusion_reaction: ReactionType,
    ion_temp_profile: NDArray[float64],
    heavier_fuel_species_fraction: float,
    fuel_ion_density_profile: NDArray[float64],
    rho: NDArray[float64],
    plasma_volume: float,
) -> tuple[float, float, float]:
    """Calculate the fusion power.

    Args:
        fusion_reaction: which nuclear reaction is being considered
        ion_temp_profile: [keV] :term:`glossary link<ion_temp_profile>`
        heavier_fuel_species_fraction: :term:`glossary link<heavier_fuel_species_fraction>`
        fuel_ion_density_profile: [1e19 m^-3] :term:`glossary link<fuel_ion_density_profile>`
        rho: [~] :term:`glossary link<rho>`
        plasma_volume: [m^3] :term:`glossary link<plasma_volume>`

    Returns:
        :term:`P_fusion` [MW], :term:`P_neutron` [MW], :term:`P_alpha` [MW]
    """

    reaction = DTFusionHively()

    power_density_factor_MW_m3 = reaction.calc_power_density(ion_temp=ion_temp_profile, heavier_fuel_species_fraction=heavier_fuel_species_fraction)
    neutral_power_density_factor_MW_m3 = reaction.calc_power_density_to_neutrals(ion_temp=ion_temp_profile, heavier_fuel_species_fraction=heavier_fuel_species_fraction)
    charged_power_density_factor_MW_m3 = reaction.calc_power_density_to_charged(ion_temp=ion_temp_profile, heavier_fuel_species_fraction=heavier_fuel_species_fraction)

    total_fusion_power_MW = _integrate_power(
        power_density_factor_MW_m3=power_density_factor_MW_m3,
        fuel_density_per_m3=fuel_ion_density_profile,
        rho=rho,
        plasma_volume=plasma_volume,
    )

    fusion_power_to_neutral_MW = _integrate_power(
        power_density_factor_MW_m3=neutral_power_density_factor_MW_m3,
        fuel_density_per_m3=fuel_ion_density_profile,
        rho=rho,
        plasma_volume=plasma_volume,
    )

    fusion_power_to_charged_MW = _integrate_power(
        power_density_factor_MW_m3=charged_power_density_factor_MW_m3,
        fuel_density_per_m3=fuel_ion_density_profile,
        rho=rho,
        plasma_volume=plasma_volume,
    )

    return total_fusion_power_MW, fusion_power_to_neutral_MW, fusion_power_to_charged_MW

@Algorithm.register_algorithm(return_keys=["neutron_power_flux_to_walls", "neutron_rate"])
def calc_neutron_flux_to_walls(
    P_neutron: float,
    surface_area: float,
    fusion_reaction: ReactionType,
    ion_temp_profile: NDArray[float64],
    heavier_fuel_species_fraction: float,
) -> tuple[float, float]:
    """Calculate the neutron loading on the wall.

    Args:
        P_neutron: [MW] :term:`glossary link<P_neutron>`
        surface_area: [m^2] :term:`glossary link<surface_area>`
        fusion_reaction: which nuclear reaction is being considered
        ion_temp_profile: [keV] :term:`glossary link<ion_temp_profile>`
        heavier_fuel_species_fraction: fraction of fuel mixture which is the heavier nuclide

    Returns:
        neutron_power_flux_to_walls [MW / m^2], neutron_rate [s^-1]
    """
    neutron_power_flux_to_walls = P_neutron / surface_area

    sigmav = SIGMAV[fusion_reaction](ion_temp_profile)
    (_, 
     rxn_energy_neut,
     _,
     _,
     _,
     _,) = ENERGY[fusion_reaction](
        sigmav=sigmav, heavier_fuel_species_fraction=heavier_fuel_species_fraction
    )

    # Prevent division by zero.
    neutron_rate = xr.where(rxn_energy_neut > 0, P_neutron / rxn_energy_neut, 0.0)

    return neutron_power_flux_to_walls, neutron_rate


def _integrate_power(
    power_density_factor_MW_m3: NDArray[float64],
    fuel_density_per_m3: NDArray[float64],
    rho: NDArray[float64],
    plasma_volume: float,
) -> float:
    """Calculate the total power due to a nuclear reaction.

    Args:
        power_density_factor_MW_m3: energy per unit volume divided by fuel species densities [MW*m^3]
        fuel_density_per_m3: density of fuel species [m^-3]
        rho: [~] :term:`glossary link<rho>`
        plasma_volume: [m^3] :term:`glossary link<plasma_volume>`

    Returns:
         power [MW]
    """
    power_density_MW_per_m3 = power_density_factor_MW_m3 * fuel_density_per_m3 * fuel_density_per_m3

    power_MW = integrate_profile_over_volume(power_density_MW_per_m3 / ureg.MW, rho, plasma_volume) * ureg.MW

    return power_MW
