"""Module to calculate the central-solenoid flux required to drive the plasma current."""

from .flux_consumption import (
    calc_breakdown_flux_consumption,
    calc_external_flux,
    calc_flux_needed_from_solenoid_over_rampup,
    calc_internal_flux,
    calc_max_flattop_duration,
    calc_poloidal_field_flux,
    calc_resistive_flux,
)
from .inductances import (
    calc_external_inductance,
    calc_internal_inductance_for_cylindrical,
    calc_internal_inductance_for_noncylindrical,
    calc_internal_inductivity,
    calc_invmu_0_dLedR,
    calc_vertical_field_mutual_inductance,
    calc_vertical_magnetic_field,
)

__all__ = [
    "calc_breakdown_flux_consumption",
    "calc_external_flux",
    "calc_external_inductance",
    "calc_flux_needed_from_solenoid_over_rampup",
    "calc_internal_flux",
    "calc_internal_inductance_for_cylindrical",
    "calc_internal_inductance_for_noncylindrical",
    "calc_internal_inductivity",
    "calc_invmu_0_dLedR",
    "calc_max_flattop_duration",
    "calc_poloidal_field_flux",
    "calc_resistive_flux",
    "calc_vertical_field_mutual_inductance",
    "calc_vertical_magnetic_field",
]
