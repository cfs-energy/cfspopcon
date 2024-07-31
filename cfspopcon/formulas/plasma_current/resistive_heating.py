"""Routines to calculate the resistivity and resistive (Ohmic) heating."""

from ...algorithm_class import Algorithm
from ...unit_handling import Unitfull, ureg, wraps_ufunc


@Algorithm.register_algorithm(return_keys=["P_ohmic"])
def calc_ohmic_power(inductive_plasma_current: Unitfull, loop_voltage: Unitfull) -> Unitfull:
    """Calculate the Ohmic heating power.

    Args:
        inductive_plasma_current: [MA] :term:`glossary link<inductive_plasma_current>`
        loop_voltage: [V] :term:`glossary link<loop_voltage>`

    Returns:
        :term:`P_ohmic` [MW]
    """
    return inductive_plasma_current * loop_voltage


@Algorithm.register_algorithm(return_keys=["spitzer_resistivity"])
@wraps_ufunc(input_units=dict(average_electron_temp=ureg.keV), return_units=dict(spitzer_resistivity=ureg.ohm * ureg.m))
def calc_Spitzer_loop_resistivity(average_electron_temp: float) -> float:
    """Calculate the parallel Spitzer loop resistivity assuming the Coulomb logarithm = 17 and Z=1.

    Resistivity from Wesson 2.16.2 :cite:`wesson_tokamaks_2011`

    Args:
        average_electron_temp: [keV] :term:`glossary link<average_electron_temp>`

    Returns:
        :term:`spitzer_resistivity` [Ohm-m]
    """
    return float((2.8e-8) * (average_electron_temp ** (-1.5)))


@Algorithm.register_algorithm(return_keys=["trapped_particle_fraction"])
def calc_resistivity_trapped_enhancement(inverse_aspect_ratio: Unitfull, resistivity_trapped_enhancement_method: int = 3) -> Unitfull:
    """Calculate the enhancement of the plasma resistivity due to trapped particles.

    Definition 1 is the denominator of eta_n (neoclassical resistivity) on p801 of Wesson :cite:`wesson_tokamaks_2011`

    Args:
        inverse_aspect_ratio: [~] :term:`glossary link<inverse_aspect_ratio>`
        resistivity_trapped_enhancement_method: [~] :term:`glossary link<resistivity_trapped_enhancement_method>`

    Returns:
        :term:`trapped_particle_fraction` [~]

    Raises:
        NotImplementedError: if resistivity_trapped_enhancement_method doesn't match an implementation
    """
    if resistivity_trapped_enhancement_method == 1:
        trapped_particle_fraction = 1 / ((1.0 - (inverse_aspect_ratio**0.5)) ** 2.0)  # pragma: nocover
    elif resistivity_trapped_enhancement_method == 2:
        trapped_particle_fraction = 2 / (1.0 - 1.31 * (inverse_aspect_ratio**0.5) + 0.46 * inverse_aspect_ratio)  # pragma: nocover
    elif resistivity_trapped_enhancement_method == 3:
        trapped_particle_fraction = 0.609 / (0.609 - 0.785 * (inverse_aspect_ratio**0.5) + 0.269 * inverse_aspect_ratio)
    else:
        raise NotImplementedError(
            f"No implementation {resistivity_trapped_enhancement_method} for calc_resistivity_trapped_enhancement."
        )  # pragma: nocover

    return trapped_particle_fraction


@Algorithm.register_algorithm(return_keys=["neoclassical_loop_resistivity"])
def calc_neoclassical_loop_resistivity(
    spitzer_resistivity: Unitfull, z_effective: Unitfull, trapped_particle_fraction: Unitfull
) -> Unitfull:
    """Calculate the neoclassical loop resistivity including impurity ions.

    Wesson Section 14.10. Impact of ion charge. Impact of dilution ~ 0.9.

    Args:
        spitzer_resistivity: [Ohm-m] :term:`glossary link<spitzer_resistivity>`
        z_effective: [~] :term:`glossary link<z_effective>`
        trapped_particle_fraction: [~] :term:`glossary link<trapped_particle_fraction>`

    Returns:
        :term:`neoclassical_loop_resistivity` [Ohm-m]
    """
    return spitzer_resistivity * z_effective * 0.9 * trapped_particle_fraction


@Algorithm.register_algorithm(return_keys=["current_relaxation_time"])
@wraps_ufunc(
    input_units=dict(
        major_radius=ureg.m,
        inverse_aspect_ratio=ureg.dimensionless,
        areal_elongation=ureg.dimensionless,
        average_electron_temp=ureg.keV,
        z_effective=ureg.dimensionless,
    ),
    return_units=dict(current_relaxation_time=ureg.s),
)
def calc_current_relaxation_time(
    major_radius: float, inverse_aspect_ratio: float, areal_elongation: float, average_electron_temp: float, z_effective: float
) -> float:
    """Calculate the current relaxation time.

    from :cite:`Bonoli`

    Args:
        major_radius: [m] :term:`glossary link<major_radius>`
        inverse_aspect_ratio: [~] :term:`glossary link<inverse_aspect_ratio>`
        areal_elongation: [~] :term:`glossary link<areal_elongation>`
        average_electron_temp: [keV] :term:`glossary link<average_electron_temp>`
        z_effective: [~] :term:`glossary link<z_effective>`

    Returns:
        :term:`current_relaxation_time` [s]
    """
    return float(
        1.4 * ((major_radius * inverse_aspect_ratio) ** 2.0) * areal_elongation * (average_electron_temp**1.5) / z_effective
    )  # [s]


@Algorithm.register_algorithm(return_keys=["loop_voltage"])
def calc_loop_voltage(
    major_radius: Unitfull,
    minor_radius: Unitfull,
    inductive_plasma_current: Unitfull,
    areal_elongation: Unitfull,
    neoclassical_loop_resistivity: Unitfull,
) -> Unitfull:
    """Calculate plasma toroidal loop voltage at flattop.

    Plasma loop voltage from Alex Creely's original work.

    Args:
        major_radius: [m] :term:`glossary link<major_radius>`
        minor_radius: [m] :term:`glossary link<minor_radius>`
        inductive_plasma_current: [MA] :term:`glossary link<inductive_plasma_current>`
        areal_elongation: [~] :term:`glossary link<areal_elongation>`
        neoclassical_loop_resistivity: [Ohm-m] :term:`glossary link<neoclassical_loop_resistivity>`

    Returns:
        :term:`loop_voltage` [V]
    """
    Iind = inductive_plasma_current  # Inductive plasma current [A]

    _term1 = 2 * major_radius / (minor_radius**2 * areal_elongation)  # Toroidal length over plasma cross-section surface area [1/m]
    return Iind * _term1 * neoclassical_loop_resistivity
