"""Calculate the plasma stored energy."""

from ...algorithm_class import Algorithm
from ...unit_handling import Unitfull, convert_units, ureg


@Algorithm.register_algorithm(return_keys=["plasma_stored_energy"])
def calc_plasma_stored_energy(
    average_electron_density: Unitfull,
    average_electron_temp: Unitfull,
    average_ion_density: Unitfull,
    summed_impurity_density: Unitfull,
    average_ion_temp: Unitfull,
    plasma_volume: Unitfull,
) -> Unitfull:
    """Calculates the plasma thermal stored energy.

    Args:
        average_electron_density: :term:`glossary link<average_electron_density>`
        average_electron_temp: :term:`glossary link<average_electron_temp>`
        average_ion_density: :term:`glossary link<average_ion_density>`
        summed_impurity_density: :term:`glossary link<summed_impurity_density>`
        average_ion_temp: :term:`glossary link<average_ion_temp>`
        plasma_volume: :term:`glossary link<plasma_volume>`

    Returns:
        :term:`plasma_stored_energy`
    """
    Wp = (
        (3.0 / 2.0)
        * (
            (average_electron_density * average_electron_temp)
            + ((average_ion_density + summed_impurity_density) * average_ion_temp)
        )
        * plasma_volume
    )

    return convert_units(Wp, ureg.MJ)
