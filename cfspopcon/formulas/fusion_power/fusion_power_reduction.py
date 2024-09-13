"""Reduce fusion power to limit by decreasing heavier fuel species fraction."""

import numpy as np
import xarray as xr
from numpy import float64
from numpy.typing import NDArray

from ...algorithm_class import Algorithm
from ...unit_handling import Unitfull


@Algorithm.register_algorithm(return_keys=["heavier_fuel_species_fraction"])
def require_P_fusion_less_than_P_fusion_limit(
    P_fusion_upper_limit: Unitfull,
    P_fusion: Unitfull,
    heavier_fuel_species_fraction: NDArray[float64],
) -> Unitfull:
    """Change heavier_fuel_species_fraction to reduce P_fusion to P_fusion_limit.

    Derived by solving
    Pfus = fT0 (1 - fT0) nD kfus
    Pmax = fT (1 - fT) nD kfus
    for fT (new_heavier_fuel_species_fraction), where fT0 is the original heavier_fuel_species_fraction, using the
    quadratic formula.

    Args:
        P_fusion_upper_limit: :term:`glossary link<P_fusion_upper_limit>`
        P_fusion: :term:`glossary link<P_fusion>`
        heavier_fuel_species_fraction: :term:`glossary link<heavier_fuel_species_fraction>`

    Returns:
        :term:`heavier_fuel_species_fraction`
    """
    # If P_fusion is less than the limit, we want the heavier_fuel_species_fraction to stay the same
    P_fusion_for_adjustment = xr.apply_ufunc(np.maximum, P_fusion, P_fusion_upper_limit)

    new_heavier_fuel_species_fraction = 0.5 - 0.5 * np.sqrt(
        1 - 4 * P_fusion_upper_limit / P_fusion_for_adjustment * heavier_fuel_species_fraction * (1 - heavier_fuel_species_fraction)
    )

    return new_heavier_fuel_species_fraction
