"""Calculate the power crossing the separatrix."""

import numpy as np

from ...algorithm_class import Algorithm
from ...unit_handling import Unitfull, ureg


@Algorithm.register_algorithm(return_keys=["power_crossing_separatrix"])
def calc_power_crossing_separatrix(P_in: Unitfull, P_radiation: Unitfull) -> Unitfull:
    """Calculate the power crossing the separatrix."""
    P_SOL = P_in - P_radiation
    return np.maximum(P_SOL, 0.0 * ureg.MW)
