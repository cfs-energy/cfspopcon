"""Routine to calculate the L-mode density limit condition function."""
from ...unit_handling import Unitfull, convert_units, ureg
from .auxillaries import calc_curvature_drive_omega_B, calc_electron_beta, calc_mass_ratio_mu
from .turbulence_wavenumbers import calc_k_EM, calc_k_RBM


def calc_L_mode_density_limit_condition(
    electron_density: Unitfull,
    electron_temperature: Unitfull,
    major_radius: Unitfull,
    magnetic_field_strength: Unitfull,
    ion_mass: Unitfull,
    alpha_c: Unitfull,
    alpha_t_turbulence_param: Unitfull,
    lambda_pe: Unitfull,
) -> Unitfull:
    """Calculate Lmode_density_limit_condition which gives the L-mode density limit at Lmode_density_limit_condition=1.

    If Lmode_density_limit_condition < 1, the operating point is stable
    If Lmode_density_limit_condition > 1, the operating point will disrupt if not above the LH transition

    Equation 3 from :cite:`Eich_2021`
    """
    omega_B = calc_curvature_drive_omega_B(lambda_perp=lambda_pe, major_radius=major_radius)
    beta_e = calc_electron_beta(
        electron_density=electron_density, electron_temperature=electron_temperature, magnetic_field_strength=magnetic_field_strength
    )
    mu = calc_mass_ratio_mu(ion_mass=ion_mass)

    k_EM = calc_k_EM(beta_e=beta_e, mu=mu)
    k_RBM = calc_k_RBM(alpha_c=alpha_c, alpha_t_turbulence_param=alpha_t_turbulence_param, omega_B=omega_B)

    return convert_units(k_EM / k_RBM, ureg.dimensionless)
