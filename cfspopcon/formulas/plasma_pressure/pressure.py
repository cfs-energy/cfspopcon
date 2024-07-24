"""Routines to calculate the pressure."""

from ...algorithm_class import Algorithm
from ...unit_handling import Unitfull, convert_units, ureg


@Algorithm.register_algorithm(return_keys=["average_total_pressure"])
def calc_average_total_pressure(
    average_electron_density: Unitfull,
    average_electron_temp: Unitfull,
    average_ion_temp: Unitfull,
) -> Unitfull:
    """Calculate the average total pressure."""
    return average_electron_density * (average_electron_temp + average_ion_temp)


@Algorithm.register_algorithm(return_keys=["peak_pressure"])
def calc_peak_pressure(
    peak_electron_temp: Unitfull,
    peak_ion_temp: Unitfull,
    peak_electron_density: Unitfull,
    peak_fuel_ion_density: Unitfull,
) -> Unitfull:
    """Calculate the peak pressure (needed for solving for the magnetic equilibrium).

    Args:
        peak_electron_temp: [keV] :term:`glossary link<peak_electron_density>`
        peak_ion_temp: [keV] :term:`glossary link<peak_ion_temp>`
        peak_electron_density: [1e19 m^-3] :term:`glossary link<peak_electron_density>`
        peak_fuel_ion_density: [~] :term:`glossary link<peak_fuel_ion_density>`

    Returns:
         peak_pressure [Pa]
    """
    return convert_units(
        peak_electron_temp * peak_electron_density
        + peak_ion_temp * peak_fuel_ion_density,
        ureg.Pa,
    )
