"""Routines to calculate divertor reattachment timescales."""

import numpy as np

from ...algorithm_class import Algorithm
from ...unit_handling import Unitfull, convert_units, ureg
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

    per [Kallenbach NME 18 (2019) 166-174]

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
        target_neutral_pressure: [Pa] :term:`glossary link<target_neutral_pressure>`
    """
    SOL_momentum_loss_fraction = calc_SOL_momentum_loss_fraction(SOL_momentum_loss_function, target_electron_temp)
    lq_int = lambda_q + 1.64 * target_gaussian_spreading
    b = lq_int / lambda_q
    q_eV = convert_units(q_parallel, "eV s**-1 m**-2")

    term1 = 3.0 / 2.0 * np.sqrt(1 - SOL_power_loss_fraction) / (1 - SOL_momentum_loss_fraction)
    term2 = np.sqrt(average_ion_mass / 2.0)
    term3 = ((2 * kappa_e0 * kappa_ez) / (7 * parallel_connection_length)) ** (2.0 / 7.0)
    term4 = (q_eV) ** (3.0 / 14.0)
    term5 = b * sheath_heat_transmission_factor * np.sin(np.radians(target_angle_of_incidence)) / (neutral_flux_density_factor * 10**23)

    p0 = (separatrix_electron_density / term1 / term2 / term3 / term4) ** 2 * term5

    return p0


@Algorithm.register_algorithm(return_keys=["reattachment_time"])
def calc_reattachment_time_henderson(
    target_neutral_pressure: Unitfull,
    target_electron_density: Unitfull,
    major_radius: Unitfull,
    parallel_connection_length: Unitfull,
    separatrix_power_transient: Unitfull,
) -> Unitfull:
    """Calculates the reattachment time for a detachment front to move to e^-5 * original front location from the target.

    Values are normalized to AUG.  per [Henderson Nucl. Fusion 64 (2024) 066006].

    Args:
      target_neutral_pressure: [Pa] :term:`glossary link<target_neutral_pressure>`
      target_electron_density: [1e19 m^-3] :term:`glossary link<target_electron_density>`
      major_radius: [m] :term:`glossary link<major_radius>`
      parallel_connection_length: [m] :term:`glossary link<parallel_connection_length>`
      separatrix_power_transient: [MW] :term:`glossary link<separatrix_power_transient>`

    Returns:
      reattachment_time: [s] :term:`glossary link<>`
    """
    term1 = target_neutral_pressure / (2.0 * ureg("Pa"))
    term2 = target_electron_density / (30.0 * ureg("n19"))
    # calculate ionization volume using AUG volume (0.4 m^3) and AUG major radius (1.65m)
    ionization_volume = major_radius / (1.65 * ureg("m")) * (0.4 * ureg("m**3"))
    term3 = ionization_volume / (0.4 * ureg("m**3"))
    term4 = parallel_connection_length / (12.0 * ureg("m"))
    term5 = 2.0 * ureg("MW") / separatrix_power_transient

    reattachment_time = 0.09 * ureg("s") * term1 * term2 * term3 * term4 * term5

    return reattachment_time
