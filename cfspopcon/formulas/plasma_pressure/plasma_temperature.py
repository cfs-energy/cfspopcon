"""Routines to calculate the plasma temperature."""
from ...algorithm_class import Algorithm

calc_average_ion_temp_from_temperature_ratio = Algorithm.from_single_function(
    lambda average_electron_temp, ion_to_electron_temp_ratio: average_electron_temp * ion_to_electron_temp_ratio,
    return_keys=["average_ion_temp"],
    name="calc_average_ion_temp_from_temperature_ratio",
)
