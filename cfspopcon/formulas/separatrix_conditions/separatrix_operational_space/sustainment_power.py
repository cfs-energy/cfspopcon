"""Routines to calculate the separatrix power required to reach the LH transition."""

import numpy as np

from ....algorithm_class import Algorithm
from ....unit_handling import Quantity, Unitfull, ureg
from .shared import calc_lambda_q_Eich2020H


@Algorithm.register_algorithm(return_keys=["sustainment_power_in_ion_channel"])
def calc_power_crossing_separatrix_in_ion_channel(
    surface_area: Unitfull,
    separatrix_electron_density: Unitfull,
    separatrix_electron_temp: Unitfull,
    alpha_t: Unitfull,
    poloidal_sound_larmor_radius: Unitfull,
    ion_heat_diffusivity: Unitfull,
    temp_scale_length_ratio: float = 1.0,
) -> Unitfull:
    """Calculate the power crossing the separatrix in the ion channel.

    This algorithm computes the power required to sustain a particular
    ion temperature gradient, given a ion heat diffusivity, using
    the method from section 4.1 in :cite:`Eich_2021`.

    temp_scale_length_ratio = Ti / Te * lambda_Te / lambda_Ti = L_Te / L_Ti

    Args:
        surface_area: :term:`glossary link<surface_area>`
        separatrix_electron_density: :term:`glossary link<separatrix_electron_density>`
        separatrix_electron_temp: :term:`glossary link<separatrix_electron_temp>`
        alpha_t: :term:`glossary link<alpha_t>`
        poloidal_sound_larmor_radius: :term:`glossary link<poloidal_sound_larmor_radius>`
        ion_heat_diffusivity: :term:`glossary link<ion_heat_diffusivity>`
        temp_scale_length_ratio: :term:`glossary link<temp_scale_length_ratio>`

    Returns:
        :term:`sustainment_power_in_ion_channel`
    """
    lambda_q = calc_lambda_q_Eich2020H(alpha_t, poloidal_sound_larmor_radius)
    lambda_Te = 3.5 * lambda_q

    L_Te = lambda_Te / separatrix_electron_temp
    L_Ti = L_Te / temp_scale_length_ratio

    P_SOL_i = surface_area * separatrix_electron_density * ion_heat_diffusivity / L_Ti

    return P_SOL_i


@Algorithm.register_algorithm(return_keys=["sustainment_power_in_electron_channel"])
def calc_power_crossing_separatrix_in_electron_channel(
    separatrix_electron_temp: Unitfull,
    target_electron_temp: Unitfull,
    cylindrical_safety_factor: Unitfull,
    major_radius: Unitfull,
    minor_radius: Unitfull,
    B_pol_out_mid: Unitfull,
    B_t_out_mid: Unitfull,
    fraction_of_P_SOL_to_divertor: Unitfull,
    z_effective: Unitfull,
    alpha_t: Unitfull,
    poloidal_sound_larmor_radius: Unitfull,
) -> Unitfull:
    """Calculate the power crossing the separatrix for a given separatrix temperature.

    Equation 11 from :cite:`Eich_2021`, inverting the Spitzer-Harm power balance.

    Args:
        separatrix_electron_temp: :term:`glossary link<separatrix_electron_temp>`
        target_electron_temp: :term:`glossary link<target_electron_temp>`
        cylindrical_safety_factor: :term:`glossary link<cylindrical_safety_factor>`
        major_radius: :term:`glossary link<major_radius>`
        minor_radius: :term:`glossary link<minor_radius>`
        B_pol_out_mid: :term:`glossary link<B_pol_out_mid>`
        B_t_out_mid: :term:`glossary link<B_t_out_mid>`
        fraction_of_P_SOL_to_divertor: :term:`glossary link<fraction_of_P_SOL_to_divertor>`
        z_effective: :term:`glossary link<z_effective>`
        alpha_t: :term:`glossary link<alpha_t>`
        poloidal_sound_larmor_radius: :term:`glossary link<poloidal_sound_larmor_radius>`

    Returns:
        :term:`sustainment_power_in_electron_channel`
    """
    lambda_q = calc_lambda_q_Eich2020H(alpha_t, poloidal_sound_larmor_radius)

    f_Zeff = 0.672 + 0.076 * np.sqrt(z_effective) + 0.252 * z_effective
    kappa_0e = Quantity(2600.0, ureg.W / (ureg.eV**3.5 * ureg.m)) / f_Zeff

    L_parallel = np.pi * cylindrical_safety_factor * major_radius

    A_SOL = 2.0 * np.pi * (major_radius + minor_radius) * lambda_q * B_pol_out_mid / B_t_out_mid

    P_SOL_e = (
        2.0
        / 7.0
        * kappa_0e
        * A_SOL
        / (L_parallel * fraction_of_P_SOL_to_divertor)
        * (separatrix_electron_temp**3.5 - target_electron_temp**3.5)
    )

    return P_SOL_e
