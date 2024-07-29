"""Calculate the Greenwald density limit."""

import numpy as np

from ...algorithm_class import Algorithm
from ...unit_handling import Unitfull, ureg, wraps_ufunc


@Algorithm.register_algorithm(return_keys=["greenwald_fraction"])
def calc_greenwald_fraction(average_electron_density: Unitfull, greenwald_density_limit: Unitfull) -> Unitfull:
    """Calculate the fraction of the Greenwald density limit.

    Args:
        average_electron_density: [1e20 m^-3] :term:`glossary link<average_electron_density>`
        greenwald_density_limit: [1e20 m^-3] :term:`glossary link<greenwald_density_limit>`

    Returns:
        :term:`greenwald_fraction` [~]
    """
    return average_electron_density / greenwald_density_limit


@Algorithm.register_algorithm(return_keys=["greenwald_density_limit"])
@wraps_ufunc(return_units=dict(nG=ureg.n20), input_units=dict(plasma_current=ureg.MA, minor_radius=ureg.m))
def calc_greenwald_density_limit(plasma_current: float, minor_radius: float) -> float:
    """Calculate the Greenwald density limit.

    Args:
        plasma_current: [MA] :term:`glossary link<plasma_current>`
        minor_radius: [m] :term:`glossary link<minor_radius>`

    Returns:
        :term:`greenwald_density_limit` [1e20 m^-3]
    """
    return plasma_current / (np.pi * minor_radius**2)


@Algorithm.register_algorithm(return_keys=["average_electron_density"])
def calc_average_electron_density_from_greenwald_fraction(greenwald_fraction: Unitfull, greenwald_density_limit: Unitfull) -> Unitfull:
    """Calculate the average electron density corresponding to a given Greenwald fraction.

    Args:
        greenwald_fraction: :term:`glossary link<greenwald_fraction>`
        greenwald_density_limit: :term:`glossary link<greenwald_density_limit>`

    Returns:
        :term:`average_electron_density` [1e20 m^-3]
    """
    return greenwald_fraction * greenwald_density_limit
