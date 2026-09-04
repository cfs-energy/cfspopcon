"""Routines to calculate the separatrix density."""

from ...algorithm_class import algorithm
from ...unit_handling import Unitfull


@algorithm(return_keys=["separatrix_electron_density"])
def calc_separatrix_electron_density(nesep_over_nebar: Unitfull, average_electron_density: Unitfull) -> Unitfull:
    """Calculate the separatrix electron density, assuming a constant ratio to the average electron density."""
    return nesep_over_nebar * average_electron_density
