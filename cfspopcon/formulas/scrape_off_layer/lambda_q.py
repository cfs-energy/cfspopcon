"""Routines to calculate the heat flux decay length (lambda_q), for several different scalings."""

from ...named_options import LambdaQScaling
from ...unit_handling import ureg, wraps_ufunc
from ...algorithm_class import Algorithm

@Algorithm.register_algorithm(return_keys=["lambda_q"])
@wraps_ufunc(
    return_units=dict(lambda_q=ureg.millimeter),
    input_units=dict(
        lambda_q_scaling=None,
        average_total_pressure=ureg.atm,
        P_sol=ureg.megawatt,
        major_radius=ureg.meter,
        B_pol_out_mid=ureg.tesla,
        inverse_aspect_ratio=ureg.dimensionless,
    ),
)
def calc_lambda_q(
    lambda_q_scaling: LambdaQScaling,
    average_total_pressure: float,
    P_sol: float,
    major_radius: float,
    B_pol_out_mid: float,
    inverse_aspect_ratio: float,
) -> float:
    """Calculate SOL heat flux decay length (lambda_q) from a scaling.

    Args:
        lambda_q_scaling: :term:`glossary link<lambda_q_scaling>`
        average_total_pressure: [atm] :term:`glossary link <average_total_pressure>`
        P_sol: [MW] :term:`glossary link<P_sol>`
        major_radius: [m] :term:`glossary link<major_radius>`
        B_pol_out_mid: [T] :term:`glossary link<B_pol_out_mid>`
        inverse_aspect_ratio: [~] :term:`glossary link<inverse_aspect_ratio>`

    Returns:
        :term:`lambda_q` [mm]
    """
    if lambda_q_scaling == LambdaQScaling.Brunner:
        return float(calc_lambda_q_with_brunner.__wrapped__(average_total_pressure))
    elif lambda_q_scaling == LambdaQScaling.EichRegression14:
        return float(calc_lambda_q_with_eich_regression_14.__wrapped__(B_pol_out_mid))
    elif lambda_q_scaling == LambdaQScaling.EichRegression15:
        return float(
            calc_lambda_q_with_eich_regression_15.__wrapped__(P_sol, major_radius, B_pol_out_mid, inverse_aspect_ratio)
        )
    else:
        raise NotImplementedError(f"No implementation for lambda_q scaling {lambda_q_scaling}")


@wraps_ufunc(
    return_units=dict(lambda_q=ureg.millimeter),
    input_units=dict(average_total_pressure=ureg.atm),
)
def calc_lambda_q_with_brunner(average_total_pressure: float) -> float:
    """Return lambda_q according to the Brunner scaling.

    Equation 4 in :cite:`brunner_2018_heat_flux`
    """
    return float(0.91 * average_total_pressure**-0.48)


@wraps_ufunc(return_units=dict(lambda_q=ureg.millimeter), input_units=dict(B_pol_out_mid=ureg.tesla))
def calc_lambda_q_with_eich_regression_14(B_pol_out_mid: float) -> float:
    """Return lambda_q according to Eich regression 14.

    #14 in Table 3 in :cite:`eich_scaling_2013`
    """
    return float(0.63 * B_pol_out_mid**-1.19)


@wraps_ufunc(
    return_units=dict(lambda_q=ureg.millimeter),
    input_units=dict(
        P_sol=ureg.megawatt,
        major_radius=ureg.meter,
        B_pol_out_mid=ureg.tesla,
        inverse_aspect_ratio=ureg.dimensionless,
    ),
)
def calc_lambda_q_with_eich_regression_15(
    P_sol: float, major_radius: float, B_pol_out_mid: float, inverse_aspect_ratio: float
) -> float:
    """Return lambda_q according to Eich regression 15.

    #15 in Table 3 in :cite:`eich_scaling_2013`
    """
    lambda_q = 1.35 * major_radius**0.04 * B_pol_out_mid**-0.92 * inverse_aspect_ratio**0.42
    if P_sol > 0:
        return float(lambda_q * P_sol**-0.02)
    else:
        return float(lambda_q)
