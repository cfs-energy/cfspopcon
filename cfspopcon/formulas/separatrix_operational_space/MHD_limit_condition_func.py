"""Routines to calculate the ideal MHD limit (using two alternative but equivalent approaches)."""
import numpy as np

from ...unit_handling import Unitfull, convert_units, ureg
from .auxillaries import calc_curvature_drive_omega_B, calc_electron_beta, calc_squared_scale_ratio_epsilon_hat
from .turbulence_wavenumbers import calc_k_ideal, calc_k_RBM


def calc_ideal_MHD_limit_condition(
    electron_density: Unitfull,
    electron_temperature: Unitfull,
    major_radius: Unitfull,
    magnetic_field_strength: Unitfull,
    safety_factor: Unitfull,
    alpha_c: Unitfull,
    alpha_t_turbulence_param: Unitfull,
    lambda_pe: Unitfull,
    ion_to_electron_temperature_ratio: float = 1.0,
) -> Unitfull:
    """Calculate c_IMHD with gives the ideal MHD limit at c_IMHD=1.

    If c_IMHD < 1, the operating point is stable
    If c_IMHD > 1, the transport will increase until c_IMHD < 1 (soft limit)

    Equation 12 from :cite:`Eich_2021`
    """
    tau_i = ion_to_electron_temperature_ratio
    k_RBM_factor = np.sqrt(2.0)

    omega_B = calc_curvature_drive_omega_B(lambda_perp=lambda_pe, major_radius=major_radius)
    beta_e = calc_electron_beta(
        electron_density=electron_density, electron_temperature=electron_temperature, magnetic_field_strength=magnetic_field_strength
    )
    epsilon_hat = calc_squared_scale_ratio_epsilon_hat(safety_factor=safety_factor, major_radius=major_radius, lambda_perp=lambda_pe)

    k_ideal = calc_k_ideal(
        beta_e=beta_e, epsilon_hat=epsilon_hat, omega_B=omega_B, tau_i=tau_i, alpha_t_turbulence_param=alpha_t_turbulence_param
    )
    k_RBM = calc_k_RBM(alpha_c=alpha_c, alpha_t_turbulence_param=alpha_t_turbulence_param, omega_B=omega_B) * k_RBM_factor

    return convert_units(k_ideal / k_RBM, ureg.dimensionless)


def calc_ideal_MHD_limit_condition_with_alpha_MHD(
    electron_density: Unitfull,
    electron_temperature: Unitfull,
    major_radius: Unitfull,
    magnetic_field_strength: Unitfull,
    safety_factor: Unitfull,
    alpha_c: Unitfull,
    lambda_pe: Unitfull,
    ion_to_electron_temperature_ratio: float = 1.0,
) -> Unitfull:
    """Calculate c_IMHD with gives the ideal MHD limit at c_IMHD=1 using alpha_MHD.

    Analytically identical to k_ideal**2 = k_RBM**2

    If c_IMHD < 1, the operating point is stable
    If c_IMHD > 1, the transport will increase until c_IMHD < 1 (soft limit)

    Equation 13 from :cite:`Eich_2021`
    """
    tau_i = ion_to_electron_temperature_ratio

    beta_e = calc_electron_beta(
        electron_density=electron_density, electron_temperature=electron_temperature, magnetic_field_strength=magnetic_field_strength
    )

    alpha_MHD = calideal_MHD_limit_condition_parameter_alpha_MHD(
        major_radius=major_radius, safety_factor=safety_factor, beta_e=beta_e, tau_i=tau_i, lambda_pe=lambda_pe
    )

    return convert_units(alpha_MHD / alpha_c, ureg.dimensionless)


def calideal_MHD_limit_condition_parameter_alpha_MHD(
    major_radius: Unitfull, safety_factor: Unitfull, beta_e: Unitfull, tau_i: Unitfull, lambda_pe: Unitfull
) -> Unitfull:
    """Calculate the MHD parameter.

    Equation 13 from :cite:`Eich_2021`
    """
    return major_radius * safety_factor**2 / lambda_pe * (1 + tau_i) * beta_e
