"""OD figures-of-merit to characterize a design point."""
import numpy as np

from ..algorithm_class import Algorithm
from ..unit_handling import Unitfull, convert_units, ureg, wraps_ufunc


@Algorithm.register_algorithm(return_keys=["fusion_triple_product"])
def calc_triple_product(peak_fuel_ion_density: Unitfull, peak_ion_temp: Unitfull, energy_confinement_time: Unitfull) -> Unitfull:
    """Calculate the fusion triple product.

    Args:
        peak_fuel_ion_density: [1e20 m^-3] :term:`glossary link<peak_fuel_ion_density>`
        peak_ion_temp: [keV] :term:`glossary link<peak_fuel_ion_density>`
        energy_confinement_time: [s] :term:`glossary link<energy_confinement_time>`

    Returns:
         fusion_triple_product [10e20 m**-3 keV s]
    """
    return peak_fuel_ion_density * peak_ion_temp * energy_confinement_time


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
    return convert_units(peak_electron_temp * peak_electron_density + peak_ion_temp * peak_fuel_ion_density, ureg.Pa)
