"""Plasma geometry (inside the last-closed-flux-surface)."""
import numpy as np

from ..unit_handling import Unitfull, ureg, wraps_ufunc


def calc_plasma_volume(major_radius: Unitfull, inverse_aspect_ratio: Unitfull, areal_elongation: Unitfull) -> Unitfull:
    """Calculate the plasma volume inside an up-down symmetrical last-closed-flux-surface.

    Geometric formulas for system codes including the effect of negative triangularity :cite: `sauter`
    NOTE: delta=1.0 is assumed since this was found to give a closer match to 2D equilibria from FreeGS.

    Args:
        major_radius: [m] :term:`glossary link<major_radius>`
        inverse_aspect_ratio: [~] :term:`glossary link<inverse_aspect_ratio>`
        areal_elongation: [~] :term:`glossary link<areal_elongation>`

    Returns:
         Vp [m^3]
    """
    return (
        2.0
        * np.pi
        * major_radius**3.0
        * inverse_aspect_ratio**2.0
        * areal_elongation
        * (np.pi - (np.pi - 8.0 / 3.0) * inverse_aspect_ratio)
    )


def calc_plasma_surface_area(major_radius: Unitfull, inverse_aspect_ratio: Unitfull, areal_elongation: Unitfull) -> Unitfull:
    """Calculate the plasma surface area inside the last-closed-flux-surface.

    Args:
        major_radius: [m] :term:`glossary link<major_radius>`
        inverse_aspect_ratio: [~] :term:`glossary link<inverse_aspect_ratio>`
        areal_elongation: [~] :term:`glossary link<areal_elongation>`

    Returns:
         Sp [m^2]
    """
    return (
        2.0 * np.pi * (major_radius**2.0) * inverse_aspect_ratio * areal_elongation * (np.pi + 2.0 - (np.pi - 2.0) * inverse_aspect_ratio)
    )


@wraps_ufunc(input_units=dict(minor_radius=ureg.m, areal_elongation=ureg.dimensionless), return_units=dict(poloidal_circumference=ureg.m))
def calc_plasma_poloidal_circumference(minor_radius: Unitfull, areal_elongation: Unitfull) -> Unitfull:
    """Calculate the plasma poloidal circumference at the last-closed-flux-surface.

    Geometric formulas for system codes including the effect of negative triangularity :cite: `sauter`

    Args:
        minor_radius: [m] :term:`glossary link<minor_radius>`
        areal_elongation: [~] :term:`glossary link<areal_elongation>`

    Returns:
         Cp [m]
    """
    return 2 * np.pi * minor_radius * (1 + 0.55 * (areal_elongation - 1))
