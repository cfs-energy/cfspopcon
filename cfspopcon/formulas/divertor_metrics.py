"""Divertor loading and functions to calculate OMP pitch (for q_parallel calculation)."""
import numpy as np
from scipy import constants  # type: ignore[import-untyped]

from ..algorithm_class import Algorithm
from ..unit_handling import ureg, wraps_ufunc


@Algorithm.register_algorithm(return_keys=["B_pol_omp"])
@wraps_ufunc(
    return_units=dict(B_pol_omp=ureg.T),
    input_units=dict(plasma_current=ureg.A, minor_radius=ureg.m),
)
def calc_B_pol_omp(plasma_current: float, minor_radius: float) -> float:
    """Calculate the poloidal magnetic field at the outboard midplane.

    Args:
        plasma_current: [MA] :term:`glossary link<plasma_current>`
        minor_radius: [m] :term:`glossary link<minor_radius>`

    Returns:
         B_pol_omp [T]
    """
    return float(constants.mu_0 * plasma_current / (2.0 * np.pi * minor_radius))


@Algorithm.register_algorithm(return_keys=["B_tor_omp"])
@wraps_ufunc(
    return_units=dict(B_tor_omp=ureg.T),
    input_units=dict(magnetic_field_on_axis=ureg.T, major_radius=ureg.m, minor_radius=ureg.m),
)
def calc_B_tor_omp(magnetic_field_on_axis: float, major_radius: float, minor_radius: float) -> float:
    """Calculate the toroidal magnetic field at the outboard midplane.

    Args:
        magnetic_field_on_axis: [T] :term:`glossary link<magnetic_field_on_axis>`
        major_radius: [m] :term:`glossary link<major_radius>`
        minor_radius: [m] :term:`glossary link<minor_radius>`

    Returns:
         B_tor_omp [T]
    """
    return magnetic_field_on_axis * (major_radius / (major_radius + minor_radius))
