"""Basic algorithms to operate on the radiated power."""
import numpy as np

from ...algorithm_class import Algorithm

calc_f_rad_core = Algorithm.from_single_function(
    func=lambda P_radiation, P_in: P_radiation / P_in, return_keys=["core_radiated_power_fraction"], name="calc_f_rad_core"
)
require_P_rad_less_than_P_in = Algorithm.from_single_function(
    lambda P_in, P_radiation: np.minimum(P_radiation, P_in), return_keys=["P_radiation"], name="require_P_rad_less_than_P_in"
)
