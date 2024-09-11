"""Routines to calculate the auxiliary (non-Ohmic, non-fusion) power."""

from ...algorithm_class import Algorithm
from ...unit_handling import Unitfull, ureg


@Algorithm.register_algorithm(return_keys=["P_external", "P_auxiliary_absorbed", "P_auxiliary_launched"])
def calc_auxiliary_power(P_in: Unitfull, P_alpha: Unitfull, P_ohmic: Unitfull, fraction_of_external_power_coupled: Unitfull) -> Unitfull:
    """Calculate the required auxiliary power.

    Args:
        P_in: [MW] :term:`glossary link<P_in>`
        P_alpha: [MW] :term:`glossary link<P_alpha>`
        P_ohmic: [MW] :term:`glossary link<P_ohmic>`
        fraction_of_external_power_coupled: [~]: :term:`glossary link<fraction_of_external_power_coupled>`

    Returns:
        :term:`P_external` [MW], :term:`P_auxiliary_absorbed` [MW], :term:`P_auxiliary_launched` [MW]
    """
    P_external = (P_in - P_alpha).clip(min=0.0 * ureg.MW)
    P_auxiliary_absorbed = (P_external - P_ohmic).clip(min=0.0 * ureg.MW)
    P_auxiliary_launched = P_auxiliary_absorbed / fraction_of_external_power_coupled

    return P_external, P_auxiliary_absorbed, P_auxiliary_launched
