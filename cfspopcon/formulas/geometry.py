"""Plasma geometry (inside the last-closed-flux-surface)."""
import numpy as np

from ..algorithm_class import Algorithm
from ..unit_handling import Unitfull


@Algorithm.register_algorithm(return_keys=["separatrix_elongation"])
def calc_separatrix_elongation_from_areal(areal_elongation: Unitfull, elongation_ratio_sep_to_areal: Unitfull) -> Unitfull:
    """Calculate the separatrix elongation from the areal elongation.

    Args:
        areal_elongation: :term:`glossary link<areal_elongation>`
        elongation_ratio_sep_to_areal: :term:`glossary link<elongation_ratio_sep_to_areal>`

    Returns:
        :term:`separatrix_elongation`
    """
    return areal_elongation * elongation_ratio_sep_to_areal


@Algorithm.register_algorithm(return_keys=["separatrix_triangularity"])
def calc_separatrix_triangularity_from_psi95(triangularity_psi95: Unitfull, triangularity_ratio_sep_to_psi95: Unitfull) -> Unitfull:
    """Calculate the separatrix triangularity from the triangularity at psiN = 0.95.

    Args:
        triangularity_psi95: :term:`glossary link<triangularity_psi95>`
        triangularity_ratio_sep_to_psi95: :term:`glossary link<triangularity_ratio_sep_to_psi95>`

    Returns:
        :term:`separatrix_triangularity`
    """
    return triangularity_psi95 * triangularity_ratio_sep_to_psi95


@Algorithm.register_algorithm(return_keys=["minor_radius"])
def calc_minor_radius_from_inverse_aspect_ratio(major_radius: Unitfull, inverse_aspect_ratio: Unitfull) -> Unitfull:
    """Calculate the minor radius from the inverse aspect ratio.

    Args:
        major_radius: :term:`glossary link<major_radius>`
        inverse_aspect_ratio: :term:`glossary link<inverse_aspect_ratio>`

    Returns:
        :term:`minor_radius`
    """
    return major_radius * inverse_aspect_ratio


@Algorithm.register_algorithm(return_keys=["vertical_minor_radius"])
def calc_vertical_minor_radius_from_elongation(minor_radius: Unitfull, separatrix_elongation: Unitfull) -> Unitfull:
    """Calculate the vertical minor radius from the elongation.

    Args:
        minor_radius: :term:`glossary link<minor_radius>`
        separatrix_elongation: :term:`glossary link<separatrix_elongation>`

    Returns:
        :term:`vertical_minor_radius`
    """
    return minor_radius * separatrix_elongation


@Algorithm.register_algorithm(return_keys=["plasma_volume"])
def calc_plasma_volume(major_radius: Unitfull, inverse_aspect_ratio: Unitfull, areal_elongation: Unitfull) -> Unitfull:
    """Calculate the plasma volume inside the last-closed-flux-surface.

    Args:
        major_radius: [m] :term:`glossary link<major_radius>`
        inverse_aspect_ratio: [~] :term:`glossary link<inverse_aspect_ratio>`
        areal_elongation: [~] :term:`glossary link<areal_elongation>`

    Returns:
         plasma_volume [m^3]
    """
    return (
        2.0
        * np.pi
        * major_radius**3.0
        * inverse_aspect_ratio**2.0
        * areal_elongation
        * (np.pi - (np.pi - 8.0 / 3.0) * inverse_aspect_ratio)
    )


@Algorithm.register_algorithm(return_keys=["surface_area"])
def calc_plasma_surface_area(major_radius: Unitfull, inverse_aspect_ratio: Unitfull, areal_elongation: Unitfull) -> Unitfull:
    """Calculate the plasma surface area inside the last-closed-flux-surface.

    Args:
        major_radius: [m] :term:`glossary link<major_radius>`
        inverse_aspect_ratio: [~] :term:`glossary link<inverse_aspect_ratio>`
        areal_elongation: [~] :term:`glossary link<areal_elongation>`

    Returns:
        surface_area [m^2]
    """
    return (
        2.0 * np.pi * (major_radius**2.0) * inverse_aspect_ratio * areal_elongation * (np.pi + 2.0 - (np.pi - 2.0) * inverse_aspect_ratio)
    )
