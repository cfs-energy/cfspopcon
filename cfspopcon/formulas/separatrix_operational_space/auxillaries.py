"""Parameters which are used to calculate the separatrix operational space, but which aren't directly outputted."""
import numpy as np

from ...unit_handling import Quantity, Unitfull, convert_units, ureg


def calc_cylindrical_edge_safety_factor(
    major_radius: Unitfull,
    minor_radius: Unitfull,
    elongation: Unitfull,
    triangularity: Unitfull,
    magnetic_field_strength: Unitfull,
    plasma_current: Unitfull,
) -> Unitfull:
    """Calculate the edge safety factor, following the formula used in the SepOS paper.

    Equation K.6 from :cite:`Eich_2021`

    Should use kappa_95 and delta_95 values.

    Gives a slightly different result to our standard q_star calculation.
    """
    shaping_correction = np.sqrt((1.0 + elongation**2 * (1.0 + 2.0 * triangularity**2 - 1.2 * triangularity**3)) / 2.0)

    poloidal_circumference = 2.0 * np.pi * minor_radius * shaping_correction

    average_B_pol = ureg.mu_0 * plasma_current / poloidal_circumference

    return magnetic_field_strength / average_B_pol * minor_radius / major_radius * shaping_correction


def calc_electron_beta(electron_density: Unitfull, electron_temperature: Unitfull, magnetic_field_strength: Unitfull) -> Unitfull:
    """Calculate the electron beta (beta_e).

    N.b. this is NOT the electron dynamical beta, which does not have the factor of 2.

    Row 5, table 2 from :cite:`Eich_2021`, corrected for factor of 2
    """
    ne = electron_density
    Te = electron_temperature
    B = magnetic_field_strength

    return 2.0 * ureg.mu_0 * ne * Te / B**2


def calc_mass_ratio_mu(ion_mass: Unitfull) -> Unitfull:
    """Calculate the electron-to-ion mass ratio.

    Row 3, table 2 from :cite:`Eich_2021`
    """
    return Quantity(1, ureg.electron_mass) / ion_mass


def calc_curvature_drive_omega_B(lambda_perp: Unitfull, major_radius: Unitfull) -> Unitfull:
    """Calculate the curvature (interchange) drive term.

    Row 6, table 2 from :cite:`Eich_2021`

    N.b. equation 7 from :cite:`Eich_2020` has an extra factor of (1 + Z)
    where Z = ne / ni where ni is summed over all charge states
    """
    return 2.0 * lambda_perp / major_radius


def calc_squared_scale_ratio_epsilon_hat(safety_factor: Unitfull, major_radius: Unitfull, lambda_perp: Unitfull) -> Unitfull:
    """Calculate the squared ratio of the parallel to perpendicular length scales.

    From text at top of right column on page 12 of :cite:`Eich_2021`
    """
    q = safety_factor
    R = major_radius

    return (q * R / lambda_perp) ** 2


def calc_critical_MHD_parameter_alpha_c(elongation: Unitfull, triangularity: Unitfull) -> Unitfull:
    """Calculate the critical value of alpha_MHD.

    Equation K.5 from :cite:`Eich_2021`
    """
    kappa = elongation
    delta = triangularity

    return kappa**1.2 * (1 + 1.5 * delta)


def calc_sound_larmor_radius_rho_s(electron_temperature: Unitfull, magnetic_field_strength: Unitfull, ion_mass: Unitfull) -> Unitfull:
    """Calculate the sound Larmor radius.

    Equation 1 from :cite:`Eich_2020`
    """
    return convert_units(np.sqrt(electron_temperature * ion_mass) / (ureg.e * magnetic_field_strength), ureg.mm)
