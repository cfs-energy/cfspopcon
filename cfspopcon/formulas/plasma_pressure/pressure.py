"""Routines to calculate the pressure."""
from ...algorithm_class import Algorithm
from ...unit_handling import Unitfull


@Algorithm.register_algorithm(return_keys=["average_total_pressure"])
def calc_average_total_pressure(
    average_electron_density: Unitfull, average_electron_temp: Unitfull, average_ion_temp: Unitfull
) -> Unitfull:
    """Calculate the average total pressure."""
    return average_electron_density * (average_electron_temp + average_ion_temp)
