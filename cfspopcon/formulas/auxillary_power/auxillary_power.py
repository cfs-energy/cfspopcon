"""Routines to calculate the auxillary (non-Ohmic, non-fusion) power."""
from ...algorithm_class import Algorithm
from ...unit_handling import Unitfull, ureg


@Algorithm.register_algorithm(return_keys=["P_auxillary"])
def calc_auxillary_power(P_external: Unitfull, P_ohmic: Unitfull) -> Unitfull:
    """Calculates the required auxillary power."""
    return (P_external - P_ohmic).clip(min=0.0 * ureg.MW)
