"""Routines to calculate divertor reattachment timescales."""

import numpy as np

from ...algorithm_class import Algorithm
from ...unit_handling import Unitfull, convert_units, magnitude_in_units, ureg
from .two_point_model.momentum_loss_functions import calc_SOL_momentum_loss_fraction


@Algorithm.register_algorithm(return_keys=["target_neutral_pressure"])
def calc_neutral_pressure_kallenbach(
    average_ion_mass: Unitfull,
    kappa_e0: Unitfull,
    kappa_ez: Unitfull,
    parallel_connection_length: Unitfull,
    target_angle_of_incidence: Unitfull,
    lambda_q: Unitfull,
    target_gaussian_spreading: Unitfull,
    sheath_heat_transmission_factor: Unitfull,
    neutral_flux_density_factor: Unitfull,
    SOL_power_loss_fraction: Unitfull,
    SOL_momentum_loss_function: Unitfull,
    separatrix_electron_density: Unitfull,
    target_electron_temp: Unitfull,
    q_parallel: Unitfull,
) -> Unitfull:
    """Calculates target neutral pressure, p0, as a function of upstream separatrix density and some other variables.

    Similar to equation 6 from :cite:`kallenbach2019neutral` and equation 2 from :cite:`henderson2024comparison`, rearranged for p0. Note that Henderson uses frad and fmom as (1 - ...) compared to Kallenbach and our definition of these terms.

    Args:
        average_ion_mass: [AMU] :term:`glossary link<average_ion_mass>`
        kappa_e0: [W m^-1 eV^(-7/2)] :term:`glossary link<kappa_e0>`
        kappa_ez: :term:`glossary link<kappa_ez>`
        parallel_connection_length: [m] :term:`glossary link<parallel_connection_length>`
        target_angle_of_incidence: [degrees] :term:`glossary link<target_angle_of_incidence>`
        lambda_q: [mm] :term:`glossary link<lambda_q>`
        target_gaussian_spreading: [mm] :term:`glossary link<target_gaussian_spreading>`
        sheath_heat_transmission_factor: :term:`glossary link<sheath_heat_transmission_factor>`
        neutral_flux_density_factor: [10^23 atoms m^-2 s^-1 Pa^-1] :term:`glossary link<neutral_flux_density_factor>`
        SOL_power_loss_fraction: :term:`glossary link<SOL_power_loss_fraction>`
        SOL_momentum_loss_function: :term:`glossary link<SOL_momentum_loss_function>`
        separatrix_electron_density: [1e19 m^-3] :term:`glossary link<separatrix_electron_density>`
        target_electron_temp: [eV] :term:`glossary link<target_electron_temp>`
        q_parallel: [GW/m²] :term:`glossary link<q_parallel>`

    Returns:
        :term:`target_neutral_pressure` [Pa]
    """
    SOL_momentum_loss_fraction = calc_SOL_momentum_loss_fraction(SOL_momentum_loss_function, target_electron_temp)
    lq_int = lambda_q + 1.64 * target_gaussian_spreading
    b = lq_int / lambda_q

    term1 = 3.0 / 2.0 * np.sqrt(1 - SOL_power_loss_fraction) / (1 - SOL_momentum_loss_fraction)
    term2 = np.sqrt(average_ion_mass / 2.0)
    term3 = ((2 * kappa_e0 * kappa_ez) / (7 * parallel_connection_length)) ** (2.0 / 7.0)
    term4 = convert_units(q_parallel, ureg.eV / (ureg.s * ureg.m**2)) ** (3.0 / 14.0)
    term5 = (
        b
        * sheath_heat_transmission_factor
        * np.sin(magnitude_in_units(target_angle_of_incidence, ureg.radian))
        / (neutral_flux_density_factor)
    )

    p0 = (separatrix_electron_density / term1 / term2 / term3 / term4) ** 2 * term5

    return p0


@Algorithm.register_algorithm(return_keys=["reattachment_time"])
def calc_reattachment_time_henderson(
    target_neutral_pressure: Unitfull,
    target_electron_density: Unitfull,
    parallel_connection_length: Unitfull,
    separatrix_power_transient: Unitfull,
    ionization_volume_density_factor: Unitfull,
    ratio_of_divertor_to_duct_pressure: Unitfull,
    ionization_volume: Unitfull,
) -> Unitfull:
    """Calculates the reattachment time for a detachment front to move to e^-5 * original front location from the target.

    Values are normalized to AUG. Equation 5 from :cite:`henderson2024comparison`

    Args:
      target_neutral_pressure: [Pa] :term:`glossary link<target_neutral_pressure>`
      target_electron_density: [1e19 m^-3] :term:`glossary link<target_electron_density>`
      parallel_connection_length: [m] :term:`glossary link<parallel_connection_length>`
      separatrix_power_transient: [MW] :term:`glossary link<separatrix_power_transient>`
      ionization_volume_density_factor: [~] :term:`glossary link<ionization_volume_density_factor>`
      ratio_of_divertor_to_duct_pressure: [~] :term:`glossary link<ratio_of_divertor_to_duct_pressure>`
      ionization_volume: [m**3] :term:`glossary link<ionization_volume>`

    Returns:
        :term:`reattachment_time` [s]
    """
    ionization_volume_average_density = ionization_volume_density_factor * target_electron_density
    term1 = target_neutral_pressure / (2.0 * ureg.Pa)
    term2 = ionization_volume_average_density / (3.0 * ureg.n20)
    term3 = ionization_volume / (0.4 * ureg.m**3)
    term4 = parallel_connection_length / (12.0 * ureg.m)
    term5 = (2.0 * ureg.MW) / separatrix_power_transient

    reattachment_time = 0.09 * ureg.s * term1 * term2 * term3 * term4 * term5 / ratio_of_divertor_to_duct_pressure

    return reattachment_time


@Algorithm.register_algorithm(return_keys=["ionization_volume"])
def calc_ionization_volume_from_AUG(
    major_radius: Unitfull,
) -> Unitfull:
    """Calculates ionization volume using major radius and AUG ionization volume.

    AUG ionization volume per :cite:`henderson2024comparison`

    Args:
      major_radius: [m] :term:`glossary link<major_radius>`

    Returns:
      :term:`ionization_volume` [m**3]
    """
    # calculate ionization volume using AUG volume (0.4 m^3) and AUG major radius (1.65m)
    ionization_volume = major_radius / (1.65 * ureg.m) * (0.4 * ureg.m**3)
    return ionization_volume


def mean_thermal_velocity(
    particle_temp: Unitfull,
    particle_mass: Unitfull,
) -> Unitfull:
    """Calculate the mean thermal velocity for thermal distribution of particles."""
    return np.sqrt(8.0 / np.pi * particle_temp / particle_mass)


@Algorithm.register_algorithm(return_keys=["neutral_flux_density_factor"])
def calc_neutral_flux_density_factor(
    average_ion_mass: Unitfull,
    ratio_of_molecular_to_ion_mass: Unitfull = 2.0,
    wall_temperature: Unitfull = 300.0 * ureg.K,
) -> Unitfull:
    """Calculate a factor to convert from a flux density to a pressure."""
    if wall_temperature.check("[temperature]"):
        wall_temperature = ureg.k_B * wall_temperature
    atoms_per_molecule = 2.0
    test_molecular_density = 1e20 * ureg.m**-3
    test_molecular_pressure = test_molecular_density * wall_temperature
    neutral_density = atoms_per_molecule * test_molecular_density
    molecular_mass = average_ion_mass * ratio_of_molecular_to_ion_mass
    onesided_maxwellian_flux_density = 0.25 * mean_thermal_velocity(wall_temperature, molecular_mass)
    flux_density_to_pascals_factor = neutral_density * onesided_maxwellian_flux_density / test_molecular_pressure
    return flux_density_to_pascals_factor
