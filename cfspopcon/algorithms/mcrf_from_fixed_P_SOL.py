import numpy as np
import xarray as xr

from .. import formulas, named_options
from ..atomic_data import read_atomic_data
from ..helpers import make_impurities_array
from ..unit_handling import Unitfull, convert_to_default_units
from .algorithm_class import Algorithm

RETURN_KEYS = [
    "minimum_core_radiated_fraction"
]

def run_calc_mcrf_from_fixed_P_SOL(P_in, P_sol_target):
    minimum_core_radiated_fraction = (P_in - P_sol_target) / P_in
    local_vars = locals()
    return {key: convert_to_default_units(local_vars[key], key) for key in RETURN_KEYS}

"""Calculate the minimum core radiated fraction from a fixed value of P_SOL (namely P_sol_target)
    Args:
        minimum_core_radiated_fraction: :term:`glossary link<minimum_core_radiated_fraction>`
        P_sol_target: :term:`glossary link<P_sol_target>`           # make a blurb?

    Returns:
        :term:`core_radiator_concentration`, :term:`P_radiated_by_core_radiator`, :term:`P_radiation`, :term:`core_radiator_concentration`, :term:`core_radiator_charge_state`, :term:`zeff_change_from_core_rad` :term:`dilution_change_from_core_rad`, :term:`z_effective`, :term:`dilution`
"""

calc_mcrf_from_fixed_P_SOL = Algorithm(
    function=run_calc_mcrf_from_fixed_P_SOL,
    return_keys=RETURN_KEYS,
)