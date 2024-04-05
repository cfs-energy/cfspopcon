"""Calculate the power due to Ohmic resistive heating."""
from .. import formulas
from ..algorithm_class import Algorithm
from ..unit_handling import Unitfull


@Algorithm.register_algorithm(
    return_keys=[
        "spitzer_resistivity",
        "trapped_particle_fraction",
        "neoclassical_loop_resistivity",
        "loop_voltage",
        "P_ohmic",
    ]
)
def calc_ohmic_power(
    bootstrap_fraction: Unitfull,
    average_electron_temp: Unitfull,
    inverse_aspect_ratio: Unitfull,
    z_effective: Unitfull,
    major_radius: Unitfull,
    minor_radius: Unitfull,
    areal_elongation: Unitfull,
    plasma_current: Unitfull,
) -> tuple[Unitfull, ...]:
    """Calculate the power due to Ohmic resistive heating.

    Args:
        bootstrap_fraction: :term:`glossary link<bootstrap_fraction>`
        average_electron_temp: :term:`glossary link<average_electron_temp>`
        inverse_aspect_ratio: :term:`glossary link<inverse_aspect_ratio>`
        z_effective: :term:`glossary link<z_effective>`
        major_radius: :term:`glossary link<major_radius>`
        minor_radius: :term:`glossary link<minor_radius>`
        areal_elongation: :term:`glossary link<areal_elongation>`
        plasma_current: :term:`glossary link<plasma_current>`

    Returns:
    :term:`spitzer_resistivity`, :term:`trapped_particle_fraction`, :term:`neoclassical_loop_resistivity`, :term:`loop_voltage`, :term:`P_ohmic`
    """
    inductive_plasma_current = plasma_current * (1.0 - bootstrap_fraction)
    spitzer_resistivity = formulas.calc_Spitzer_loop_resistivity(average_electron_temp)
    trapped_particle_fraction = formulas.calc_resistivity_trapped_enhancement(inverse_aspect_ratio)
    neoclassical_loop_resistivity = formulas.calc_neoclassical_loop_resistivity(spitzer_resistivity, z_effective, trapped_particle_fraction)
    loop_voltage = formulas.calc_loop_voltage(
        major_radius, minor_radius, inductive_plasma_current, areal_elongation, neoclassical_loop_resistivity
    )
    P_ohmic = formulas.calc_ohmic_power(inductive_plasma_current, loop_voltage)

    return (spitzer_resistivity, trapped_particle_fraction, neoclassical_loop_resistivity, loop_voltage, P_ohmic)
