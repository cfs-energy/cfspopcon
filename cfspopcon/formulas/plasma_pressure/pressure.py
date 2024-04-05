"""Routines to calculate the pressure."""
from ...algorithm_class import Algorithm


@Algorithm.register_algorithm(return_keys=["average_total_pressure"])
def calc_average_total_pressure(average_electron_density, average_electron_temp, average_ion_temp):
    """Calculate the average total pressure."""
    return average_electron_density * (average_electron_temp + average_ion_temp)
