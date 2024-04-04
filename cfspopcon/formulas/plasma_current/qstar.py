"""Routines relating the plasma current to an analytical estimate of the 95% safety factor qstar."""
from ...algorithm_class import Algorithm
from ...unit_handling import Unitfull, ureg, wraps_ufunc


@Algorithm.register_algorithm(return_keys=["f_shaping"])
def calc_f_shaping_for_qstar(inverse_aspect_ratio: Unitfull, areal_elongation: Unitfull, triangularity_psi95: Unitfull) -> Unitfull:
    """Calculate the shaping function.

    Equation A11 from ITER Physics Basis Ch. 1. Eqn. A-11 :cite:`editors_iter_1999`
    See following discussion for how this function is used.
    q_95 = 5 * minor_radius^2 * magnetic_field_on_axis / (R * plasma_current) f_shaping

    Args:
        inverse_aspect_ratio: [~] :term:`glossary link<inverse_aspect_ratio>`
        areal_elongation: [~] :term:`glossary link<areal_elongation>`
        triangularity_psi95: [~] :term:`glossary link<triangularity_psi95>`

    Returns:
        :term:`f_shaping` [~]
    """
    return ((1.0 + areal_elongation**2.0 * (1.0 + 2.0 * triangularity_psi95**2.0 - 1.2 * triangularity_psi95**3.0)) / 2.0) * (
        (1.17 - 0.65 * inverse_aspect_ratio) / (1.0 - inverse_aspect_ratio**2.0) ** 2.0
    )


@Algorithm.register_algorithm(return_keys=["plasma_current"])
@wraps_ufunc(
    input_units=dict(
        magnetic_field_on_axis=ureg.T,
        major_radius=ureg.m,
        inverse_aspect_ratio=ureg.dimensionless,
        q_star=ureg.dimensionless,
        f_shaping=ureg.dimensionless,
    ),
    return_units=dict(plasma_current=ureg.MA),
)
def calc_plasma_current_from_qstar(
    magnetic_field_on_axis: float, major_radius: float, inverse_aspect_ratio: float, q_star: float, f_shaping: float
) -> float:
    """Calculate the plasma current in mega-amperes.

    Updated formula from ITER Physics Basis Ch. 1. :cite:`editors_iter_1999`

    Args:
        magnetic_field_on_axis: [T] :term:`glossary link<magnetic_field_on_axis>`
        major_radius: [m] :term:`glossary link<major_radius>`
        inverse_aspect_ratio: [~] :term:`glossary link<inverse_aspect_ratio>`
        q_star: [~] :term:`glossary link<q_star>`
        f_shaping: [~] :term:`glossary link<f_shaping>`

    Returns:
        :term:`plasma_current` [MA]
    """
    return (  # type:ignore[no-any-return]
        5.0 * ((inverse_aspect_ratio * major_radius) ** 2.0) * (magnetic_field_on_axis / (q_star * major_radius)) * f_shaping
    )


@Algorithm.register_algorithm(return_keys=["q_star"])
@wraps_ufunc(
    input_units=dict(
        magnetic_field_on_axis=ureg.T,
        major_radius=ureg.m,
        inverse_aspect_ratio=ureg.dimensionless,
        plasma_current=ureg.MA,
        f_shaping=ureg.dimensionless,
    ),
    return_units=dict(q_star=ureg.dimensionless),
)
def calc_q_star_from_plasma_current(
    magnetic_field_on_axis: float, major_radius: float, inverse_aspect_ratio: float, plasma_current: float, f_shaping: float
) -> float:
    """Calculate an analytical estimate for the edge safety factor q_star.

    Updated formula from ITER Physics Basis Ch. 1. :cite:`editors_iter_1999`

    Args:
        magnetic_field_on_axis: [T] :term:`glossary link<magnetic_field_on_axis>`
        major_radius: [m] :term:`glossary link<major_radius>`
        inverse_aspect_ratio: [~] :term:`glossary link<inverse_aspect_ratio>`
        plasma_current: [MA] :term:`glossary link<plasma_current>`
        f_shaping: [~] :term:`glossary link<f_shaping>`

    Returns:
        :term:`qstar` [~]
    """
    return (  # type:ignore[no-any-return]
        5.0 * (inverse_aspect_ratio * major_radius) ** 2.0 * magnetic_field_on_axis / (plasma_current * major_radius) * f_shaping
    )
