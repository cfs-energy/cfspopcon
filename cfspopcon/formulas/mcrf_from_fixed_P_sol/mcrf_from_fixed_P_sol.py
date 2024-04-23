from ...unit_handling import convert_to_default_units
from ...algorithm_class import Algorithm

@Algorithm.register_algorithm(return_keys=["minimum_core_radiated_fraction"])
def calc_mcrf_from_fixed_P_sol(P_in, P_sol_target):
    """Calculate the minimum core radiated fraction from a fixed value of P_SOL (i.e. P_sol_target)

    Args:
        P_in: :term:`glossary link<P_in>`
        P_sol_target: :term:`glossary link<P_sol_target>`           # make a blurb?

    Returns:
        minimum_core_radiated_fraction: :term:`glossary link<minimum_core_radiated_fraction>`
    """
    return  (P_in - P_sol_target) / P_in