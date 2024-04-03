"""Operational limits to avoid disruptive regions."""
import numpy as np

from ..algorithm_class import Algorithm
from ..unit_handling import ureg, wraps_ufunc


@Algorithm.register_algorithm(return_keys=["greenwald_fraction"])
@wraps_ufunc(
    return_units=dict(greenwald_fraction=ureg.dimensionless),
    input_units=dict(
        average_electron_density=ureg.n20, inverse_aspect_ratio=ureg.dimensionless, major_radius=ureg.m, plasma_current=ureg.MA
    ),
)
def calc_greenwald_fraction(
    average_electron_density: float, inverse_aspect_ratio: float, major_radius: float, plasma_current: float
) -> float:
    """Calculate the fraction of the Greenwald density limit.

    Args:
        average_electron_density: [1e20 m^-3] :term:`glossary link<average_electron_density>`
        inverse_aspect_ratio: [~] :term:`glossary link<inverse_aspect_ratio>`
        major_radius: [m] :term:`glossary link<major_radius>`
        plasma_current: [MA] :term:`glossary link<plasma_current>`

    Returns:
        :term:`greenwald_fraction` [~]
    """
    n_Greenwald = calc_greenwald_density_limit.__wrapped__(plasma_current, inverse_aspect_ratio * major_radius)

    return float(average_electron_density / n_Greenwald)


@Algorithm.register_algorithm(return_keys=["greenwald_density_limit"])
@wraps_ufunc(return_units=dict(nG=ureg.n20), input_units=dict(plasma_current=ureg.MA, minor_radius=ureg.m))
def calc_greenwald_density_limit(plasma_current: float, minor_radius: float) -> float:
    """Calculate the Greenwald density limit.

    Args:
        plasma_current: [MA] :term:`glossary link<plasma_current>`
        minor_radius: [m] :term:`glossary link<minor_radius>`

    Returns:
        nG Greenwald density limit [n20]
    """
    return plasma_current / (np.pi * minor_radius**2)


@Algorithm.register_algorithm(return_keys=["troyon_max_beta"])
@wraps_ufunc(
    return_units=dict(troyon_max_beta=ureg.percent),
    input_units=dict(minor_radius=ureg.m, magnetic_field_on_axis=ureg.T, plasma_current=ureg.MA),
)
def calc_troyon_limit(minor_radius: float, magnetic_field_on_axis: float, plasma_current: float) -> float:
    """Calculate the maximum value for beta, according to the Troyon limit.

    Args:
        minor_radius: [m] :term:`glossary link<minor_radius>`
        magnetic_field_on_axis: [T] :term:`glossary link<magnetic_field_on_axis>`
        plasma_current: [MA] :term:`glossary link<plasma_current>`

    Returns:
         troyon_max_beta [~]
    """
    return 2.8 * plasma_current / (minor_radius * magnetic_field_on_axis)
