"""Calculate the normalized collisionality."""
import numpy as np

from ...algorithm_class import Algorithm
from ...unit_handling import Unitfull, convert_units, ureg, wraps_ufunc


@wraps_ufunc(input_units=dict(ne=ureg.m**-3, Te=ureg.eV), return_units=dict(Lambda_c=ureg.dimensionless))
def calc_coulomb_logarithm(ne: float, Te: float) -> float:
    """Calculate the Coulomb logarithm, for electron-electron or electron-ion collisions.

    From text on page 6 of :cite:`Verdoolaege_2021`
    """
    return float(30.9 - np.log(ne**0.5 * Te**-1.0))


@Algorithm.register_algorithm(return_keys=["nu_star"])
def calc_normalised_collisionality(
    average_electron_density: Unitfull,
    average_electron_temp: Unitfull,
    average_ion_temp: Unitfull,
    q_star: Unitfull,
    major_radius: Unitfull,
    inverse_aspect_ratio: Unitfull,
    z_effective: Unitfull,
) -> Unitfull:
    """Calculate normalized collisionality.

    Equation 1c from :cite:`Verdoolaege_2021`

    Extra factor of ureg.e**2, presumably related to Te**-2 for Te in eV

    Args:
        average_electron_density: [1e19 m^-3] :term:`glossary link<average_electron_density>`
        average_electron_temp: [keV] :term:`glossary link<average_electron_temp>`
        average_ion_temp: [keV] :term:`glossary link<average_ion_temp>`
        q_star: [~] :term:`glossary link<q_star>`
        major_radius: [m] :term:`glossary link<major_radius>`
        inverse_aspect_ratio: [m] :term:`glossary link<inverse_aspect_ratio>`
        z_effective: [~] :term:`glossary link<z_effective>`

    Returns:
         nu_star [~]
    """
    return convert_units(
        ureg.e**4
        / (2.0 * np.pi * 3**1.5 * ureg.epsilon_0**2)
        * calc_coulomb_logarithm(average_electron_density, average_electron_temp)
        * average_electron_density
        * q_star
        * major_radius
        * z_effective
        / (average_ion_temp**2 * inverse_aspect_ratio**1.5),
        ureg.dimensionless,
    )
