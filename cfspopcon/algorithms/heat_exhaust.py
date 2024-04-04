"""Calculate the parallel heat flux density upstream and related metrics."""
import numpy as np

from .. import formulas, named_options
from ..algorithm_class import Algorithm
from ..unit_handling import Unitfull, ureg


@Algorithm.register_algorithm(
    return_keys=[
        "PB_over_R",
        "PBpRnSq",
        "B_pol_out_mid",
        "B_t_out_mid",
        "fieldline_pitch_at_omp",
        "lambda_q",
        "q_parallel",
        "q_perp",
    ]
)
def calc_heat_exhaust(
    P_sol: Unitfull,
    magnetic_field_on_axis: Unitfull,
    major_radius: Unitfull,
    inverse_aspect_ratio: Unitfull,
    plasma_current: Unitfull,
    minor_radius: Unitfull,
    q_star: Unitfull,
    average_electron_density: Unitfull,
    average_total_pressure: Unitfull,
    fraction_of_P_SOL_to_divertor: Unitfull,
    lambda_q_scaling: named_options.LambdaQScaling,
    lambda_q_factor: Unitfull = 1.0 * ureg.dimensionless,
) -> dict[str, Unitfull]:
    """Calculate the parallel heat flux density upstream and related metrics.

    Args:
        P_sol: :term:`glossary link<P_sol>`
        magnetic_field_on_axis: :term:`glossary link<magnetic_field_on_axis>`
        major_radius: :term:`glossary link<major_radius>`
        inverse_aspect_ratio: :term:`glossary link<inverse_aspect_ratio>`
        plasma_current: :term:`glossary link<plasma_current>`
        minor_radius: :term:`glossary link<minor_radius>`
        q_star: :term:`glossary link<q_star>`
        average_electron_density: :term:`glossary link<average_electron_density>`
        average_total_pressure: :term:`glossary link <average_total_pressure>`
        fraction_of_P_SOL_to_divertor: :term:`glossary link <fraction_of_P_SOL_to_divertor>`
        lambda_q_scaling: :term:`glossary link<lambda_q_scaling>`
        lambda_q_factor: :term:`glossary link<lambda_q_factor>`

    Returns:
        :term:`PB_over_R`, :term:`PBpRnSq`, :term:`B_pol_out_mid`, :term:`B_t_out_mid`, :term:`fieldline_pitch_at_omp`, :term:`lambda_q`, :term:`q_parallel`, :term:`q_perp`

    """
    PB_over_R = P_sol * magnetic_field_on_axis / major_radius
    PBpRnSq = (P_sol * (magnetic_field_on_axis / q_star) / major_radius) / (average_electron_density**2.0)

    B_pol_out_mid = formulas.calc_B_pol_omp(plasma_current=plasma_current, minor_radius=minor_radius)
    B_t_out_mid = formulas.calc_B_tor_omp(magnetic_field_on_axis, major_radius, minor_radius)

    fieldline_pitch_at_omp = np.sqrt(B_t_out_mid**2 + B_pol_out_mid**2) / B_pol_out_mid

    lambda_q = lambda_q_factor * formulas.scrape_off_layer_model.calc_lambda_q(
        lambda_q_scaling, average_total_pressure, P_sol, major_radius, B_pol_out_mid, inverse_aspect_ratio
    )

    q_parallel = formulas.scrape_off_layer_model.calc_parallel_heat_flux_density(
        P_sol, fraction_of_P_SOL_to_divertor, major_radius + minor_radius, lambda_q, fieldline_pitch_at_omp
    )
    q_perp = P_sol / (2.0 * np.pi * (major_radius + minor_radius) * lambda_q)

    return (PB_over_R, PBpRnSq, B_pol_out_mid, B_t_out_mid, fieldline_pitch_at_omp, lambda_q, q_parallel, q_perp)
