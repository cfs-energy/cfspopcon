"""Estimate the temperature peaking."""

from ...algorithm_class import Algorithm
from ...unit_handling import Unitfull


@Algorithm.register_algorithm(return_keys=["peak_electron_temp", "peak_ion_temp"])
def calc_temperature_peaking(
    average_electron_temp: Unitfull,
    average_ion_temp: Unitfull,
    temperature_peaking: Unitfull,
) -> tuple[Unitfull, ...]:
    """Apply the temperature peaking.

    Args:
        average_electron_temp: :term:`glossary link<average_electron_temp>`
        average_ion_temp: :term:`glossary link<average_ion_temp>`
        temperature_peaking: :term:`glossary link<temperature_peaking>`

    Returns:
        :term:`peak_electron_temp`, :term:`peak_ion_temp`
    """
    peak_electron_temp = average_electron_temp * temperature_peaking
    peak_ion_temp = average_ion_temp * temperature_peaking

    return peak_electron_temp, peak_ion_temp
