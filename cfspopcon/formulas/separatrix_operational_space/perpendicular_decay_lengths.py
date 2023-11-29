"""Routines to calculate perpendicular decay lengths used in the separatrix operational space."""
from ...unit_handling import Unitfull, ureg, wraps_ufunc


def calc_lambda_pe_Eich2021H(alpha_t_turbulence_param: Unitfull, rho_s_pol: Unitfull, factor: float = 3.6) -> Unitfull:
    """Calculate the H-mode electron pressure decay length.

    Equation K.1 from :cite:`Eich_2021`
    """
    return 1.2 * (1 + factor * alpha_t_turbulence_param**1.9) * rho_s_pol


@wraps_ufunc(input_units=dict(alpha_t_turbulence_param=ureg.dimensionless), return_units=dict(lambda_pe=ureg.mm))
def calc_lambda_pe_Manz2023L(alpha_t_turbulence_param: float) -> float:
    """Calculate the L-mode electron pressure decay length.

    Equation B.1 from :cite:`Manz_2023`
    """
    return 17.3 * alpha_t_turbulence_param**0.298  # type:ignore[no-any-return]


def calc_lambda_q_Eich2020H(alpha_t_turbulence_param: Unitfull, rho_s_pol: Unitfull) -> Unitfull:
    """Calculate the H-mode heat flux decay length."""
    lambda_Te = 2.1 * (1 + 2.1 * alpha_t_turbulence_param**1.7) * rho_s_pol
    return 2.0 / 7.0 * lambda_Te
