"""Routine to calculate the separatrix power required to sustain a given point.

Closely related to scrape_off_layer_model/separatrix_electron_temp.py, but with additional corrections.
"""
import numpy as np

from ...unit_handling import Quantity, Unitfull, convert_units, ureg


def calc_power_crossing_separatrix(
    separatrix_temp: Unitfull,
    target_temp: Unitfull,
    cylindrical_edge_safety_factor: Unitfull,
    major_radius: Unitfull,
    minor_radius: Unitfull,
    lambda_q: Unitfull,
    B_pol_omp: Unitfull,
    B_tor_omp: Unitfull,
    f_share: Unitfull,
    Zeff: Unitfull,
) -> Unitfull:
    """Calculate the power crossing the separatrix for a given separatrix temperature."""
    f_Zeff = 0.672 + 0.076 * np.sqrt(Zeff) + 0.252 * Zeff
    kappa_0e = Quantity(2600.0, ureg.W / (ureg.eV**3.5 * ureg.m)) / f_Zeff

    L_parallel = np.pi * cylindrical_edge_safety_factor * major_radius

    A_SOL = 2.0 * np.pi * (major_radius + minor_radius) * lambda_q * B_pol_omp / B_tor_omp

    return convert_units(2.0 / 7.0 * kappa_0e * A_SOL / (L_parallel * f_share) * (separatrix_temp**3.5 - target_temp**3.5), ureg.MW)
