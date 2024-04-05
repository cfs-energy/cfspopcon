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


@Algorithm.register_algorithm(return_keys=["P_fusion", "P_neutron", "P_alpha"])
@wraps_ufunc(
    return_units=dict(P_fusion=ureg.MW, P_neutron=ureg.MW, P_alpha=ureg.MW),
    input_units=dict(
        fusion_reaction=None,
        ion_temp_profile=ureg.keV,
        heavier_fuel_species_fraction=ureg.dimensionless,
        fuel_ion_density_profile=ureg.n19,
        rho=ureg.dimensionless,
        plasma_volume=ureg.m**3,
    ),
    input_core_dims=[(), ("dim_rho",), (), ("dim_rho",), ("dim_rho",), ()],
    output_core_dims=[(), (), ()],
)
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
    reaction_at_Ti = _calc_fusion_reaction_rate(fusion_reaction, ion_temp_profile, heavier_fuel_species_fraction)

    power_density_factor_MW_m3 = reaction_at_Ti[4]
    neutral_power_density_factor_MW_m3 = reaction_at_Ti[5]
    charged_power_density_factor_MW_m3 = reaction_at_Ti[6]

    total_fusion_power_MW = _integrate_power(
        power_density_factor_MW_m3=power_density_factor_MW_m3,
        fuel_density_per_m3=fuel_ion_density_profile * 1e19,
        rho=rho,
        plasma_volume=plasma_volume,
    )

    fusion_power_to_neutral_MW = _integrate_power(
        power_density_factor_MW_m3=neutral_power_density_factor_MW_m3,
        fuel_density_per_m3=fuel_ion_density_profile * 1e19,
        rho=rho,
        plasma_volume=plasma_volume,
    )

    fusion_power_to_charged_MW = _integrate_power(
        power_density_factor_MW_m3=charged_power_density_factor_MW_m3,
        fuel_density_per_m3=fuel_ion_density_profile * 1e19,
        rho=rho,
        plasma_volume=plasma_volume,
    )

    return total_fusion_power_MW, fusion_power_to_neutral_MW, fusion_power_to_charged_MW

@Algorithm.register_algorithm(return_keys=["neutron_power_flux_to_walls", "neutron_rate"])
@wraps_ufunc(
    return_units=dict(neutron_power_flux_to_walls=ureg.MW / ureg.m**2, neutron_rate=ureg.s**-1),
    input_units=dict(
        P_neutron=ureg.MW,
        surface_area=ureg.m**2,
        fusion_reaction=None,
        ion_temp_profile=ureg.keV,
        heavier_fuel_species_fraction=ureg.dimensionless,
    ),
    input_core_dims=[(), (), (), ("dim_rho",), ()],
    output_core_dims=[(), ()],
)
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
    rxn_energy_neut = _calc_fusion_reaction_rate(fusion_reaction, ion_temp_profile, heavier_fuel_species_fraction)[2]
    if rxn_energy_neut > 0:  # This will happen for D-He3 reactions
        neutron_rate = P_neutron / rxn_energy_neut  # [MW / MJ] -> [1 / s]
    else:
        neutron_rate = 0.0

    return neutron_power_flux_to_walls, neutron_rate


@wraps_ufunc(
    return_units=dict(
        sigmav=ureg.cm**3 / ureg.s,
        rxn_energy=ureg.MJ,
        rxn_energy_neut=ureg.MJ,
        rxn_energy_charged=ureg.MJ,
        number_power_dens=ureg.MW * ureg.m**3,
        number_power_dens_neut=ureg.MW * ureg.m**3,
        number_power_dens_charged=ureg.MW * ureg.m**3,
    ),
    input_units=dict(
        fusion_reaction=None,
        ion_temp_profile=ureg.keV,
        heavier_fuel_species_fraction=ureg.dimensionless,
    ),
    input_core_dims=[(), ("dim_rho",), ()],
    output_core_dims=[("dim_rho",), (), (), (), ("dim_rho",), ("dim_rho",), ("dim_rho",)],
)
def calc_fusion_reaction_rate(
    fusion_reaction: ReactionType, ion_temp_profile: NDArray[float64], heavier_fuel_species_fraction: float
) -> tuple[
    NDArray[float64],
    Union[NDArray[float64], float],
    float,
    Union[NDArray[float64], float],
    NDArray[float64],
    NDArray[float64],
    NDArray[float64],
]:
    """Calculate reaction properties based on reaction type, mixture ratio, and temperature.

    Args:
        fusion_reaction: which nuclear reaction is being considered
        ion_temp_profile: [keV] :term:`glossary link<ion_temp_profile>`
        heavier_fuel_species_fraction: fraction of fuel mixture which is the heavier nuclide

    Returns:
        :A tuple holding:

        :sigmav: rate coefficient <sigma*v> for the given ion temperature [cm^3/s]
        :rxn_energy: total energy released per reaction [MJ]
        :rxn_energy_neut: energy released to neutral products per reaction [MJ]
        :rxn_energy_charged: energy released to charged products per reaction [MJ]
        :number_power_dens: power per unit volume divided by reactant densities [MW*m^3]
        :number_power_dens_neut: power per unit volume divided by reactant densities deposited in neutral products [MW*m^3]
        :number_power_dens_charged: power per unit volume divided by reactant densities deposited in charged products [MW*m^3]
    """
    return _calc_fusion_reaction_rate(fusion_reaction, ion_temp_profile, heavier_fuel_species_fraction)


def _calc_fusion_reaction_rate(
    fusion_reaction: ReactionType, ion_temp_profile: NDArray[float64], heavier_fuel_species_fraction: float
) -> tuple[
    NDArray[float64],
    Union[NDArray[float64], float],
    float,
    Union[NDArray[float64], float],
    NDArray[float64],
    NDArray[float64],
    NDArray[float64],
]:
    """Calculate reaction properties based on reaction type, mixture ratio, and temperature, without unit-handling.

    Args:
        fusion_reaction: which nuclear reaction is being considered
        ion_temp_profile: [keV] :term:`glossary link<ion_temp_profile>`
        heavier_fuel_species_fraction: fraction of fuel mixture which is the heavier nuclide

    Returns:
        :A tuple holding:

        :sigmav: rate coefficient <sigma*v> for the given ion temperature [cm^3/s]
        :rxn_energy: total energy released per reaction [MJ]
        :rxn_energy_neut: energy released to neutral products per reaction [MJ]
        :rxn_energy_charged: energy released to charged products per reaction [MJ]
        :number_power_dens: power per unit volume divided by reactant densities [MW*m^3]
        :number_power_dens_neut: power per unit volume divided by reactant densities deposited in neutral products [MW*m^3]
        :number_power_dens_charged: power per unit volume divided by reactant densities deposited in charged products [MW*m^3]
    """
    sigmav_func = SIGMAV[fusion_reaction]  # Reaction rate function to use based on reaction type
    energy_func = ENERGY[fusion_reaction]  # Reaction energy function to use based on reaction type

    sigmav = sigmav_func(ion_temp_profile)

    # This generates a false positive when type checking, as the type checker doesn't
    # realize that the sigmav_func and energy_func pair always correctly matches.
    # That's because the return type of a dictionary can't be narrowed based on a runtime key.
    (rxn_energy, rxn_energy_neut, rxn_energy_charged, number_power_dens, number_power_dens_neut, number_power_dens_charged,) = energy_func(
        sigmav=sigmav, heavier_fuel_species_fraction=heavier_fuel_species_fraction  # type:ignore[call-arg]
    )

    return (
        sigmav,
        rxn_energy,
        rxn_energy_neut,
        rxn_energy_charged,
        number_power_dens,
        number_power_dens_neut,
        number_power_dens_charged,
    )  # type:ignore[return-value]


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
    power_MW = integrate_profile_over_volume(power_density_MW_per_m3, rho, plasma_volume)

    return power_MW
