"""Calculate rho_star, which gives the radio of the device size to the Larmor radius."""
import numpy as np

from ...algorithm_class import Algorithm
from ...unit_handling import Unitfull, convert_units, ureg


@Algorithm.register_algorithm(return_keys=["rho_star"])
def calc_rho_star(
    fuel_average_mass_number: Unitfull, average_ion_temp: Unitfull, magnetic_field_on_axis: Unitfull, minor_radius: Unitfull
) -> Unitfull:
    """Calculate rho* (normalized gyroradius).

    Equation 1a from :cite:`Verdoolaege_2021`

    Args:
        fuel_average_mass_number: [amu] :term:`glossary link<fuel_average_mass_number>`
        average_ion_temp: [keV] :term:`glossary link<average_ion_temp>`
        magnetic_field_on_axis: :term:`glossary link<magnetic_field_on_axis>`
        minor_radius: [m] :term:`glossary link<minor_radius>`

    Returns:
         rho_star [~]
    """
    return convert_units(
        np.sqrt(fuel_average_mass_number * average_ion_temp) / (ureg.e * magnetic_field_on_axis * minor_radius), ureg.dimensionless
    )
