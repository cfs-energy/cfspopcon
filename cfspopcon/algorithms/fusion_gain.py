"""Calculate the fusion power and thermal gain (Q)."""
from .. import formulas, named_options
from ..algorithm_class import Algorithm
from ..unit_handling import Unitfull, ureg


@Algorithm.register_algorithm(
    return_keys=[
        "P_fusion",
        "P_neutron",
        "P_alpha",
        "P_external",
        "P_launched",
        "Q",
        "neutron_power_flux_to_walls",
        "neutron_rate",
    ]
)
def calc_fusion_gain(
    fusion_reaction: named_options.ReactionType,
    ion_temp_profile: Unitfull,
    heavier_fuel_species_fraction: Unitfull,
    fuel_ion_density_profile: Unitfull,
    rho: Unitfull,
    plasma_volume: Unitfull,
    surface_area: Unitfull,
    P_in: Unitfull,
    fraction_of_external_power_coupled: Unitfull,
) -> tuple[Unitfull, ...]:
    """Calculate the fusion power and thermal gain (Q).

    Args:
        fusion_reaction: :term:`glossary link<fusion_reaction>`
        ion_temp_profile: :term:`glossary link<ion_temp_profile>`
        heavier_fuel_species_fraction: :term:`glossary link<heavier_fuel_species_fraction>`
        fuel_ion_density_profile: :term:`glossary link<fuel_ion_density_profile>`
        rho: :term:`glossary link<rho>`
        plasma_volume: :term:`glossary link<plasma_volume>`
        surface_area: :term:`glossary link<surface_area>`
        P_in: :term:`glossary link<P_in>`
        fraction_of_external_power_coupled: :term:`glossary link<fraction_of_external_power_coupled>`

    Returns:
        :term:`P_fusion`, :term:`P_neutron`, :term:`P_alpha`, :term:`P_external`, :term:`P_launched`, :term:`Q`, :term:`neutron_power_flux_to_walls` :term:`neutron_rate`
    """
    P_fusion, P_neutron, P_alpha = formulas.calc_fusion_power(
        fusion_reaction, ion_temp_profile, heavier_fuel_species_fraction, fuel_ion_density_profile, rho, plasma_volume
    )

    P_external = (P_in - P_alpha).clip(min=0.0 * ureg.MW)
    P_launched = P_external / fraction_of_external_power_coupled
    Q = formulas.thermal_calc_gain_factor(P_fusion, P_launched)

    neutron_power_flux_to_walls, neutron_rate = formulas.calc_neutron_flux_to_walls(
        P_neutron, surface_area, fusion_reaction, ion_temp_profile, heavier_fuel_species_fraction
    )

    return (P_fusion, P_neutron, P_alpha, P_external, P_launched, Q, neutron_power_flux_to_walls, neutron_rate)
