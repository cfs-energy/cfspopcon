"""Routines to calculate the upstream (unmitigated) parallel heat flux density."""
from numpy import pi

from ...unit_handling import Unitfull


def calc_parallel_heat_flux_density(
    power_crossing_separatrix: Unitfull,
    fraction_of_P_SOL_to_divertor: Unitfull,
    upstream_major_radius: Unitfull,
    lambda_q: Unitfull,
    upstream_fieldline_pitch: Unitfull,
) -> Unitfull:
    """Calculate the parallel heat flux density entering the flux tube (q_par).

    This expression is power to target divided by the area perpendicular to the flux tube.
    1. Power to target = power crossing separatrix * fraction of that power going to the target considered
    2. The poloidal area of a ring at the outboard midplane is 2 * pi * (R + minor_radius) * width
    3. For the width, we take the heat flux decay length lambda_q
    4. P_SOL * f_share / (2 * pi * (R + minor_radius) lambda_q) gives the heat flux per poloidal area
    5. We project this poloidal heat flux density into a parallel heat flux density by dividing by the field-line pitch

    Args:
      power_crossing_separatrix: [MW] :term:`glossary link<power_crossing_separatrix>`
      fraction_of_P_SOL_to_divertor: :term:`glossary link <fraction_of_P_SOL_to_divertor>`
      upstream_major_radius: [m] R + minor_radius, major radius at outboard midplane separatrix
      lambda_q: [mm] :term:`glossary link<lambda_q>`
      upstream_fieldline_pitch: B_total / B_poloidal at outboard midplane separatrix [~]

    Returns:
      q_parallel [GW/m^2]
    """
    return (
        power_crossing_separatrix * fraction_of_P_SOL_to_divertor / (2.0 * pi * upstream_major_radius * lambda_q) * upstream_fieldline_pitch
    )
