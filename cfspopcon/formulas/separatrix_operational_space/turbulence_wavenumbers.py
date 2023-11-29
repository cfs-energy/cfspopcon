"""Routines to calculate turbulence wavenumbers."""
import numpy as np

from ...unit_handling import Unitfull


def calc_k_ideal(
    beta_e: Unitfull, epsilon_hat: Unitfull, omega_B: Unitfull, tau_i: Unitfull, alpha_t_turbulence_param: Unitfull
) -> Unitfull:
    """Calculate k_ideal, which gives the spatial scale of ideal MHD modes.

    Equation G.3 from :cite:`Eich_2021`.

    N.b. G.3 is written in terms of beta_hat = beta_e * epsilon_hat
    where epsilon_hat = (q R / lambda_pe)**2
    """
    return np.sqrt(beta_e * epsilon_hat * omega_B**1.5 * (1.0 + tau_i) / alpha_t_turbulence_param)


def calc_k_RBM(alpha_c: Unitfull, alpha_t_turbulence_param: Unitfull, omega_B: Unitfull) -> Unitfull:
    """Calculate k_RBM, which gives the spatial scale of resistive ballooning modes.

    Equation B.12 from :cite:`Eich_2021`

    N.b. There is an extra factor of Lambda_pe multiplied on omega_B compared to
    table 1. In Appendix A it is noted that Lambda_pe = lambda_perp / lambda_pe = 1.
    """
    return np.sqrt(alpha_c / alpha_t_turbulence_param * np.sqrt(omega_B))


def calc_k_EM(beta_e: Unitfull, mu: Unitfull) -> Unitfull:
    """Calculate k_EM, which gives the spatial scale of electromagnetic effects.

    Equation D.3 from :cite:`Eich_2021`.
    """
    return np.sqrt(beta_e / mu)
