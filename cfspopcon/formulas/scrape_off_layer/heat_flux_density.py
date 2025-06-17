"""Calculate the parallel and perpendicular heat flux density entering the scrape-off-layer."""

import numpy as np
from scipy import constants

from ...algorithm_class import Algorithm
from ...unit_handling import Unitfull, ureg, wraps_ufunc


@Algorithm.register_algorithm(return_keys=["B_pol_out_mid"])
@wraps_ufunc(
    return_units=dict(B_pol_omp=ureg.T),
    input_units=dict(plasma_current=ureg.A, minor_radius=ureg.m),
)
def calc_B_pol_omp(plasma_current: float, minor_radius: float) -> float:
    """Calculate the poloidal magnetic field at the outboard midplane.

    Args:
        plasma_current: [MA] :term:`glossary link<plasma_current>`
        minor_radius: [m] :term:`glossary link<minor_radius>`

    Returns:
         B_pol_out_mid [T]
    """
    return float(constants.mu_0 * plasma_current / (2.0 * np.pi * minor_radius))


@Algorithm.register_algorithm(return_keys=["B_t_out_mid"])
def calc_B_tor_omp(magnetic_field_on_axis: Unitfull, major_radius: Unitfull, minor_radius: Unitfull) -> Unitfull:
    """Calculate the toroidal magnetic field at the outboard midplane.

    Args:
        magnetic_field_on_axis: [T] :term:`glossary link<magnetic_field_on_axis>`
        major_radius: [m] :term:`glossary link<major_radius>`
        minor_radius: [m] :term:`glossary link<minor_radius>`

    Returns:
         B_t_out_mid [T]
    """
    return magnetic_field_on_axis * (major_radius / (major_radius + minor_radius))


@Algorithm.register_algorithm(return_keys=["fieldline_pitch_at_omp"])
def calc_fieldline_pitch_at_omp(B_t_out_mid: Unitfull, B_pol_out_mid: Unitfull) -> Unitfull:
    """Calculate the pitch of the magnetic field at the outboard midplane.

    Args:
        B_t_out_mid: [T] :term:`glossary link<B_t_out_mid>`
        B_pol_out_mid: [T] :term:`glossary link<B_pol_out_mid>`

    Returns:
         fieldline_pitch_at_omp [~]
    """
    return np.sqrt(B_t_out_mid**2 + B_pol_out_mid**2) / B_pol_out_mid


@Algorithm.register_algorithm(return_keys=["q_parallel"])
def calc_parallel_heat_flux_density(
    power_crossing_separatrix: Unitfull,
    fraction_of_P_SOL_to_divertor: Unitfull,
    major_radius: Unitfull,
    minor_radius: Unitfull,
    lambda_q: Unitfull,
    fieldline_pitch_at_omp: Unitfull,
) -> Unitfull:
    """Calculate the parallel heat flux density entering a flux tube (q_par) at the outboard midplane.

    This expression is power to target divided by the area perpendicular to the flux tube.
    1. Power to target = power crossing separatrix * fraction of that power going to the target considered
    2. The poloidal area of a ring at the outboard midplane is 2 * pi * (R + minor_radius) * width
    3. For the width, we take the heat flux decay length lambda_q
    4. P_SOL * f_share / (2 * pi * (R + minor_radius) lambda_q) gives the heat flux per poloidal area
    5. We project this poloidal heat flux density into a parallel heat flux density by dividing by the field-line pitch

    Args:
      power_crossing_separatrix: [MW] :term:`glossary link<power_crossing_separatrix>`
      fraction_of_P_SOL_to_divertor: :term:`glossary link <fraction_of_P_SOL_to_divertor>`
      major_radius: [m] :term:`glossary link <major_radius>`
      minor_radius: [m] :term:`glossary link <minor_radius>`
      lambda_q: [mm] :term:`glossary link<lambda_q>`
      fieldline_pitch_at_omp: B_total / B_poloidal at outboard midplane separatrix [~]

    Returns:
      q_parallel [GW/m^2]
    """
    upstream_major_radius = major_radius + minor_radius
    return (
        power_crossing_separatrix
        * fraction_of_P_SOL_to_divertor
        / (2.0 * np.pi * upstream_major_radius * lambda_q)
        * fieldline_pitch_at_omp
    )


@Algorithm.register_algorithm(return_keys=["q_perp"])
def calc_q_perp(power_crossing_separatrix: Unitfull, major_radius: Unitfull, minor_radius: Unitfull, lambda_q: Unitfull) -> Unitfull:
    """Calculate the perpendicular heat flux at the outboard midplane.

    Args:
      power_crossing_separatrix: [MW] :term:`glossary link<power_crossing_separatrix>`
      major_radius: [m] :term:`glossary link <major_radius>`
      minor_radius: [m] :term:`glossary link <minor_radius>`
      lambda_q: [mm] :term:`glossary link<lambda_q>`

    Returns:
      q_perp [MW/m^2]
    """
    return power_crossing_separatrix / (2.0 * np.pi * (major_radius + minor_radius) * lambda_q)
