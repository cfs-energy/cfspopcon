"""Routines to calculate the plasma current, safety factor and ohmic heating."""

from .bootstrap_fraction import calc_bootstrap_fraction, calc_inductive_plasma_current
from .flux_consumption import (
    calc_breakdown_flux_consumption,
    calc_external_flux,
    calc_external_inductance,
    calc_flux_needed_from_solenoid_over_rampup,
    calc_internal_flux,
    calc_internal_inductance_for_cylindrical,
    calc_internal_inductance_for_noncylindrical,
    calc_internal_inductivity,
    calc_invmu_0_dLedR,
    calc_max_flattop_duration,
    calc_poloidal_field_flux,
    calc_resistive_flux,
    calc_vertical_field_mutual_inductance,
    calc_vertical_magnetic_field,
)
from .resistive_heating import (
    calc_current_relaxation_time,
    calc_loop_voltage,
    calc_neoclassical_loop_resistivity,
    calc_ohmic_power,
    calc_resistivity_trapped_enhancement,
    calc_Spitzer_loop_resistivity,
)
from .safety_factor import (
    calc_cylindrical_edge_safety_factor,
    calc_f_shaping_for_qstar,
    calc_plasma_current_from_qstar,
    calc_q_star_from_plasma_current,
)

__all__ = [
    "calc_bootstrap_fraction",
    "calc_inductive_plasma_current",
    "calc_f_shaping_for_qstar",
    "calc_plasma_current_from_qstar",
    "calc_q_star_from_plasma_current",
    "calc_current_relaxation_time",
    "calc_loop_voltage",
    "calc_neoclassical_loop_resistivity",
    "calc_ohmic_power",
    "calc_resistivity_trapped_enhancement",
    "calc_Spitzer_loop_resistivity",
    "calc_cylindrical_edge_safety_factor",
    "calc_internal_flux",
    "calc_external_flux",
    "calc_resistive_flux",
    "calc_poloidal_field_flux",
    "calc_flux_needed_from_solenoid_over_rampup",
    "calc_max_flattop_duration",
    "calc_breakdown_flux_consumption",
    "calc_internal_inductivity",
    "calc_internal_inductance_for_cylindrical",
    "calc_internal_inductance_for_noncylindrical",
    "calc_external_inductance",
    "calc_vertical_field_mutual_inductance",
    "calc_invmu_0_dLedR",
    "calc_vertical_magnetic_field",
]
