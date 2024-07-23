"""Interface to different energy confinement scalings and routines to calculate the plasma stored energy."""

from .plasma_stored_energy import calc_plasma_stored_energy
from .read_energy_confinement_scalings import ConfinementScaling, read_confinement_scalings
from .solve_for_input_power import solve_tau_e_scaling_for_input_power
from .switch_confinement_scaling_on_threshold import (
    switch_to_linearised_ohmic_confinement_below_threshold,
    switch_to_L_mode_confinement_below_threshold,
)

__all__ = [
    "calc_plasma_stored_energy",
    "read_confinement_scalings",
    "ConfinementScaling",
    "solve_tau_e_scaling_for_input_power",
    "switch_to_linearised_ohmic_confinement_below_threshold",
    "switch_to_L_mode_confinement_below_threshold",
]
