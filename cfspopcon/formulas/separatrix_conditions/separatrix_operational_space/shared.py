"""Shared functions used for computing the separatrix operational space."""

import numpy as np

from ....algorithm_class import Algorithm
from ....unit_handling import Quantity, Unitfull, ureg, wraps_ufunc
from ...metrics.larmor_radius import calc_larmor_radius


def calc_ideal_MHD_wavenumber(
    beta_e: Unitfull,
    epsilon_hat: Unitfull,
    omega_B: Unitfull,
    tau_i: Unitfull,
    alpha_t: Unitfull,
) -> Unitfull:
    """Calculate k_ideal, which gives the spatial scale of ideal MHD modes.

    Equation G.3 from :cite:`Eich_2021`.

    N.b. G.3 is written in terms of beta_hat = beta_e * epsilon_hat
    where epsilon_hat = (q R / electron_pressure_decay_length)**2
    """
    return np.sqrt(beta_e * epsilon_hat * omega_B**1.5 * (1.0 + tau_i) / alpha_t)


def calc_resistive_ballooning_wavenumber(
    critical_alpha_MHD: Unitfull, alpha_t: Unitfull, omega_B: Unitfull
) -> Unitfull:
    """Calculate k_RBM, which gives the spatial scale of resistive ballooning modes.

    This is the square root of the right-hand-side of equation 4 from :cite:`Eich_2021`.

    From equation D.3 we see that the left-hand-side of equation 4 is k_EM^2.
    In G.3 this is related to k_ideal. Equation 4 is given as equation 3 (k_ideal = k_RBM),
    so if the left-hand-side is ~k_ideal^2, the right-hand-side is ~k_RBM^2. N.b. this isn't
    entirely clear from the paper, but it does reproduce the results from the paper.

    N.b. There is an extra factor of Lambda_pe multiplied on omega_B compared to
    table 1. In Appendix A it is noted that Lambda_pe = perpendicular_decay_length / electron_pressure_decay_length = 1.
    """
    return np.sqrt(critical_alpha_MHD / alpha_t * np.sqrt(omega_B))


def calc_electromagnetic_wavenumber(beta_e: Unitfull, mu: Unitfull) -> Unitfull:
    """Calculate k_EM, which gives the spatial scale of electromagnetic effects.

    Equation D.3 from :cite:`Eich_2021`, but without a factor of 2 since we are
    using the electron beta and not the electron dynamical beta.
    """
    return np.sqrt(beta_e / mu)


def calc_electron_beta(
    electron_density: Unitfull,
    electron_temp: Unitfull,
    magnetic_field_strength: Unitfull,
) -> Unitfull:
    """Calculate the electron beta (beta_e).

    N.b. this is NOT the electron dynamical beta, which does not have the factor of 2.

    Row 5, table 2 from :cite:`Eich_2021`, corrected for factor of 2
    """
    return (
        2.0
        * ureg.mu_0
        * electron_density
        * electron_temp
        / magnetic_field_strength**2
    )


def calc_electron_to_ion_mass_ratio(ion_mass: Unitfull) -> Unitfull:
    """Calculate the electron-to-ion mass ratio.

    Row 3, table 2 from :cite:`Eich_2021`
    """
    return Quantity(1, ureg.electron_mass) / ion_mass


def calc_curvature_drive(
    perpendicular_decay_length: Unitfull, major_radius: Unitfull
) -> Unitfull:
    """Calculate the curvature (interchange) drive term.

    Row 6, table 2 from :cite:`Eich_2021`

    N.b. equation 7 from :cite:`Eich_2020` has an extra factor of (1 + Z)
    where Z = ne / ni where ni is summed over all charge states
    """
    return 2.0 * perpendicular_decay_length / major_radius


def calc_squared_scale_ratio(
    safety_factor: Unitfull,
    major_radius: Unitfull,
    perpendicular_decay_length: Unitfull,
) -> Unitfull:
    """Calculate the squared ratio of the parallel to perpendicular length scales.

    From text at top of right column on page 12 of :cite:`Eich_2021`
    """
    return (safety_factor * major_radius / perpendicular_decay_length) ** 2


@Algorithm.register_algorithm(return_keys=["critical_alpha_MHD"])
def calc_critical_alpha_MHD(
    elongation_psi95: Unitfull, triangularity_psi95: Unitfull
) -> Unitfull:
    """Calculate the critical value of alpha_MHD.

    Equation K.5 from :cite:`Eich_2021`
    """
    return elongation_psi95**1.2 * (1 + 1.5 * triangularity_psi95)


@Algorithm.register_algorithm(return_keys=["poloidal_sound_larmor_radius"])
def calc_poloidal_sound_larmor_radius(
    minor_radius: Unitfull,
    elongation_psi95: Unitfull,
    triangularity_psi95: Unitfull,
    plasma_current: Unitfull,
    separatrix_electron_temp: Unitfull,
    ion_mass: Unitfull,
) -> Unitfull:
    """Calculate the poloidally-averaged sound Larmor radius (rho_s_pol).

    Equation K.4 from :cite:`Eich_2021`, using B_pol from equations K.6 and K.7.

    Args:
        minor_radius: :term:`glossary link<minor_radius>`
        elongation_psi95: :term:`glossary link<elongation_psi95>`
        triangularity_psi95: :term:`glossary link<triangularity_psi95>`
        plasma_current: :term:`glossary link<plasma_current>`
        separatrix_electron_temp: :term:`glossary link<separatrix_electron_temp>`
        ion_mass: :term:`glossary link<ion_mass>`

    Returns:
        :term:`poloidal_sound_larmor_radius`
    """
    poloidal_circumference = (
        2.0
        * np.pi
        * minor_radius
        * (1 + 0.55 * (elongation_psi95 - 1))
        * (1 + 0.08 * triangularity_psi95**2)
    )
    B_pol_avg = ureg.mu_0 * plasma_current / poloidal_circumference
    return calc_larmor_radius(
        species_temperature=separatrix_electron_temp,
        magnetic_field_strength=B_pol_avg,
        species_mass=ion_mass,
    )


def calc_electron_pressure_decay_length_Eich2021H(
    alpha_t: Unitfull, poloidal_sound_larmor_radius: Unitfull, factor: float = 3.6
) -> Unitfull:
    """Calculate the H-mode electron pressure decay length.

    Equation K.1 from :cite:`Eich_2021`
    """
    return 1.2 * (1 + factor * alpha_t**1.9) * poloidal_sound_larmor_radius


@wraps_ufunc(
    input_units=dict(alpha_t=ureg.dimensionless),
    return_units=dict(electron_pressure_decay_length=ureg.mm),
)
def calc_electron_pressure_decay_length_Manz2023L(alpha_t: float) -> float:
    """Calculate the L-mode electron pressure decay length.

    Equation B.1 from :cite:`Manz_2023`
    """
    return 17.3 * alpha_t**0.298  # type:ignore[no-any-return]


def calc_lambda_q_Eich2020H(
    alpha_t: Unitfull, poloidal_sound_larmor_radius: Unitfull
) -> Unitfull:
    """Calculate the H-mode heat flux decay length.

    Equation 22 from :cite:`Eich_2020`, then using lambda_q = 2/7 lambda_Te (equation A1)
    """
    lambda_Te = 2.1 * (1 + 2.1 * alpha_t**1.7) * poloidal_sound_larmor_radius
    return 2.0 / 7.0 * lambda_Te
