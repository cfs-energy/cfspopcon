"""Routines to calculate the heat flux decay length (lambda_q), for several different scalings."""

from ...algorithm_class import Algorithm
from ...named_options import LambdaQScaling
from ...unit_handling import ureg, wraps_ufunc


@Algorithm.register_algorithm(return_keys=["lambda_q"])
@wraps_ufunc(
    return_units=dict(lambda_q=ureg.millimeter),
    input_units=dict(
        lambda_q_scaling=None,
        average_total_pressure=ureg.atm,
        power_crossing_separatrix=ureg.megawatt,
        major_radius=ureg.meter,
        B_pol_out_mid=ureg.tesla,
        inverse_aspect_ratio=ureg.dimensionless,
        lambda_q_factor=ureg.dimensionless,
    ),
)
def calc_lambda_q(
    lambda_q_scaling: LambdaQScaling,
    average_total_pressure: float,
    power_crossing_separatrix: float,
    major_radius: float,
    B_pol_out_mid: float,
    inverse_aspect_ratio: float,
    lambda_q_factor: float = 1.0,
) -> float:
    """Calculate SOL heat flux decay length (lambda_q) from a scaling.

    TODO: Remove in next major release, in favour of using algorithms directly.

    Args:
        lambda_q_scaling: :term:`glossary link<lambda_q_scaling>`
        average_total_pressure: [atm] :term:`glossary link <average_total_pressure>`
        power_crossing_separatrix: [MW] :term:`glossary link<power_crossing_separatrix>`
        major_radius: [m] :term:`glossary link<major_radius>`
        B_pol_out_mid: [T] :term:`glossary link<B_pol_out_mid>`
        inverse_aspect_ratio: [~] :term:`glossary link<inverse_aspect_ratio>`
        lambda_q_factor: [~] :term:`glossary link<lambda_q_factor>`

    Returns:
        :term:`lambda_q` [mm]
    """
    if lambda_q_scaling == LambdaQScaling.Brunner:
        return float(
            calc_lambda_q_with_brunner.unitless_func(average_total_pressure=average_total_pressure, lambda_q_factor=lambda_q_factor)
        )
    elif lambda_q_scaling == LambdaQScaling.EichRegression14:
        return float(calc_lambda_q_with_eich_regression_14.unitless_func(B_pol_out_mid=B_pol_out_mid, lambda_q_factor=lambda_q_factor))
    elif lambda_q_scaling == LambdaQScaling.EichRegression15:
        return float(
            calc_lambda_q_with_eich_regression_15.unitless_func(
                power_crossing_separatrix=power_crossing_separatrix,
                major_radius=major_radius,
                B_pol_out_mid=B_pol_out_mid,
                inverse_aspect_ratio=inverse_aspect_ratio,
                lambda_q_factor=lambda_q_factor,
            )
        )
    else:
        raise NotImplementedError(f"No implementation for lambda_q scaling {lambda_q_scaling}")


@Algorithm.register_algorithm(return_keys=["lambda_q"])
@wraps_ufunc(
    return_units=dict(lambda_q=ureg.millimeter),
    input_units=dict(average_total_pressure=ureg.atm, lambda_q_factor=ureg.dimensionless),
)
def calc_lambda_q_with_brunner(average_total_pressure: float, lambda_q_factor: float = 1.0) -> float:
    """Return lambda_q according to the Brunner scaling.

    Equation 4 in :cite:`brunner_2018_heat_flux`
    """
    return float(lambda_q_factor * 0.91 * average_total_pressure**-0.48)


@Algorithm.register_algorithm(return_keys=["lambda_q"])
@wraps_ufunc(
    return_units=dict(lambda_q=ureg.millimeter),
    input_units=dict(
        magnetic_field_on_axis=ureg.T,
        q_star=ureg.dimensionless,
        power_crossing_separatrix=ureg.megawatt,
        lambda_q_factor=ureg.dimensionless,
    ),
)
def calc_lambda_q_with_eich_regression_9(
    magnetic_field_on_axis: float, q_star: float, power_crossing_separatrix: float, lambda_q_factor: float = 1.0
) -> float:
    """Return lambda_q according to Eich regression 9.

    #9 in Table 2 in :cite:`eich_scaling_2013`
    """
    return float(lambda_q_factor * 0.7 * magnetic_field_on_axis**-0.77 * q_star**1.05 * power_crossing_separatrix**0.09)


@wraps_ufunc(return_units=dict(lambda_q=ureg.millimeter), input_units=dict(B_pol_out_mid=ureg.tesla, lambda_q_factor=ureg.dimensionless))
def calc_lambda_q_with_eich_regression_14(B_pol_out_mid: float, lambda_q_factor: float = 1.0) -> float:
    """Return lambda_q according to Eich regression 14.

    #14 in Table 3 in :cite:`eich_scaling_2013`
    """
    return float(lambda_q_factor * 0.63 * B_pol_out_mid**-1.19)


@Algorithm.register_algorithm(return_keys=["lambda_q"])
@wraps_ufunc(
    return_units=dict(lambda_q=ureg.millimeter),
    input_units=dict(
        power_crossing_separatrix=ureg.megawatt,
        major_radius=ureg.meter,
        B_pol_out_mid=ureg.tesla,
        inverse_aspect_ratio=ureg.dimensionless,
        lambda_q_factor=ureg.dimensionless,
    ),
)
def calc_lambda_q_with_eich_regression_15(
    power_crossing_separatrix: float, major_radius: float, B_pol_out_mid: float, inverse_aspect_ratio: float, lambda_q_factor: float = 1.0
) -> float:
    """Return lambda_q according to Eich regression 15.

    #15 in Table 3 in :cite:`eich_scaling_2013`
    """
    lambda_q = 1.35 * major_radius**0.04 * B_pol_out_mid**-0.92 * inverse_aspect_ratio**0.42
    if power_crossing_separatrix > 0:
        return float(lambda_q_factor * lambda_q * power_crossing_separatrix**-0.02)
    else:
        return float(lambda_q_factor * lambda_q)
