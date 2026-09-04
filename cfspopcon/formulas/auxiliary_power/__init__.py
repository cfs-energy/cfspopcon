"""Routines to calculate the auxiliary (non-Ohmic, non-fusion) power."""

from .auxiliary_power import calc_auxiliary_power, calc_input_power_for_fixed_auxiliary_power

__all__ = ["calc_auxiliary_power", "calc_input_power_for_fixed_auxiliary_power"]
