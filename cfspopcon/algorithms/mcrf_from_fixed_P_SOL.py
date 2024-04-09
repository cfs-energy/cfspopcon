from ..unit_handling import convert_to_default_units
from .algorithm_class import Algorithm

RETURN_KEYS = [
    "minimum_core_radiated_fraction", 
]

def run_calc_mcrf_from_fixed_P_sol(P_in, P_sol_target):
    minimum_core_radiated_fraction = (P_in - P_sol_target) / P_in
    local_vars = locals()
    return {key: convert_to_default_units(local_vars[key], key) for key in RETURN_KEYS}

    """Calculate the minimum core radiated fraction from a fixed value of P_SOL (i.e. P_sol_target)

    Args:
        P_in: :term:`glossary link<P_in>`
        P_sol_target: :term:`glossary link<P_sol_target>`           # make a blurb?

    Returns:
        minimum_core_radiated_fraction: :term:`glossary link<minimum_core_radiated_fraction>`
    """

calc_mcrf_from_fixed_P_sol = Algorithm(
    function=run_calc_mcrf_from_fixed_P_sol,
    return_keys=RETURN_KEYS,
)