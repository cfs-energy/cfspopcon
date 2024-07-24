"""Plasma geometry (inside the last-closed-flux-surface)."""

import numpy as np

from ...algorithm_class import Algorithm
from ...unit_handling import Unitfull


@Algorithm.register_algorithm(return_keys=["plasma_volume"])
def calc_plasma_volume(
    major_radius: Unitfull, inverse_aspect_ratio: Unitfull, areal_elongation: Unitfull
) -> Unitfull:
    """Calculate the plasma volume inside an up-down symmetrical last-closed-flux-surface.

    Geometric formulas for system codes including the effect of negative triangularity :cite: `sauter`
    NOTE: delta=1.0 is assumed since this was found to give a closer match to 2D equilibria from FreeGS.

    Args:
        major_radius: [m] :term:`glossary link<major_radius>`
        inverse_aspect_ratio: [~] :term:`glossary link<inverse_aspect_ratio>`
        areal_elongation: [~] :term:`glossary link<areal_elongation>`

    Returns:
        :term:`plasma_volume` [m^3]
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
def calc_plasma_surface_area(
    major_radius: Unitfull, inverse_aspect_ratio: Unitfull, areal_elongation: Unitfull
) -> Unitfull:
    """Calculate the plasma surface area inside the last-closed-flux-surface.

    Args:
        major_radius: [m] :term:`glossary link<major_radius>`
        inverse_aspect_ratio: [~] :term:`glossary link<inverse_aspect_ratio>`
        areal_elongation: [~] :term:`glossary link<areal_elongation>`

    Returns:
        :term:`surface_area` [m^2]
    """
    return (
        2.0
        * np.pi
        * (major_radius**2.0)
        * inverse_aspect_ratio
        * areal_elongation
        * (np.pi + 2.0 - (np.pi - 2.0) * inverse_aspect_ratio)
    )


@Algorithm.register_algorithm(return_keys=["poloidal_circumference"])
def calc_plasma_poloidal_circumference(
    minor_radius: Unitfull, areal_elongation: Unitfull
) -> Unitfull:
    """Calculate the plasma poloidal circumference at the last-closed-flux-surface.

    Geometric formulas for system codes including the effect of negative triangularity :cite: `sauter`

    Args:
        minor_radius: [m] :term:`glossary link<minor_radius>`
        areal_elongation: [~] :term:`glossary link<areal_elongation>`

    Returns:
        :term:`poloidal_circumference` [m]
    """
    return 2 * np.pi * minor_radius * (1 + 0.55 * (areal_elongation - 1))


calc_areal_elongation_from_elongation_at_psi95 = Algorithm.from_single_function(
    func=lambda elongation_psi95, elongation_ratio_areal_to_psi95: elongation_psi95
    * elongation_ratio_areal_to_psi95,
    return_keys=["areal_elongation"],
    name="calc_areal_elongation_from_elongation_at_psi95",
)

calc_elongation_at_psi95_from_areal_elongation = Algorithm.from_single_function(
    func=lambda areal_elongation, elongation_ratio_areal_to_psi95: areal_elongation
    / elongation_ratio_areal_to_psi95,
    return_keys=["elongation_psi95"],
    name="calc_elongation_at_psi95_from_areal_elongation",
)

calc_separatrix_elongation_from_areal_elongation = Algorithm.from_single_function(
    func=lambda areal_elongation, elongation_ratio_sep_to_areal: areal_elongation
    * elongation_ratio_sep_to_areal,
    return_keys=["separatrix_elongation"],
    name="calc_separatrix_elongation_from_areal_elongation",
)

calc_separatrix_triangularity_from_triangularity95 = Algorithm.from_single_function(
    func=lambda triangularity_psi95, triangularity_ratio_sep_to_psi95: triangularity_psi95
    * triangularity_ratio_sep_to_psi95,
    return_keys=["separatrix_triangularity"],
    name="calc_separatrix_triangularity_from_triangularity95",
)

calc_minor_radius_from_inverse_aspect_ratio = Algorithm.from_single_function(
    func=lambda major_radius, inverse_aspect_ratio: major_radius * inverse_aspect_ratio,
    return_keys=["minor_radius"],
    name="calc_minor_radius_from_inverse_aspect_ratio",
)

calc_inverse_aspect_ratio = Algorithm.from_single_function(
    func=lambda major_radius, minor_radius: minor_radius / major_radius,
    return_keys=["inverse_aspect_ratio"],
    name="calc_inverse_aspect_ratio",
)

calc_vertical_minor_radius_from_elongation_and_minor_radius = (
    Algorithm.from_single_function(
        func=lambda minor_radius, separatrix_elongation: minor_radius
        * separatrix_elongation,
        return_keys=["vertical_minor_radius"],
        name="calc_vertical_minor_radius_from_elongation_and_minor_radius",
    )
)
