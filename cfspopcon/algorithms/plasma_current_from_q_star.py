"""Calculate plasma current from edge safety factor."""
from .. import formulas
from ..algorithm_class import Algorithm
from ..unit_handling import Unitfull


@Algorithm.register_algorithm(
    return_keys=[
        "f_shaping",
        "plasma_current",
    ]
)
def calc_plasma_current_from_q_star(
    magnetic_field_on_axis: Unitfull,
    major_radius: Unitfull,
    q_star: Unitfull,
    inverse_aspect_ratio: Unitfull,
    areal_elongation: Unitfull,
    triangularity_psi95: Unitfull,
) -> dict[str, Unitfull]:
    """Calculate plasma current from edge safety factor.

    Args:
        magnetic_field_on_axis: :term:`glossary link<magnetic_field_on_axis>`
        major_radius: :term:`glossary link<major_radius>`
        q_star: :term:`glossary link<q_star>`
        inverse_aspect_ratio: :term:`glossary link<inverse_aspect_ratio>`
        areal_elongation: :term:`glossary link<areal_elongation>`
        triangularity_psi95: :term:`glossary link<triangularity_psi95>`

    Returns:
    term:`f_shaping`, term:`plasma_current`,

    """
    f_shaping = formulas.calc_f_shaping(inverse_aspect_ratio, areal_elongation, triangularity_psi95)
    plasma_current = formulas.calc_plasma_current(magnetic_field_on_axis, major_radius, inverse_aspect_ratio, q_star, f_shaping)

    return (f_shaping, plasma_current)
