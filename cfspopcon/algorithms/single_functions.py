"""Algorithm wrappers for single functions which don't fit into larger algorithms."""
import numpy as np

from .. import deprecated_formulas
from ..algorithm_class import Algorithm

calc_f_rad_core = Algorithm.from_single_function(
    func=lambda P_radiation, P_in: P_radiation / P_in, return_keys=["core_radiated_power_fraction"], name="calc_f_rad_core"
)
calc_normalised_collisionality = Algorithm.from_single_function(
    func=deprecated_formulas.calc_normalised_collisionality, return_keys=["nu_star"], name="calc_normalised_collisionality"
)
calc_rho_star = Algorithm.from_single_function(func=deprecated_formulas.calc_rho_star, return_keys=["rho_star"], name="calc_rho_star")
calc_triple_product = Algorithm.from_single_function(
    func=deprecated_formulas.calc_triple_product, return_keys=["fusion_triple_product"], name="calc_triple_product"
)
calc_peak_pressure = Algorithm.from_single_function(
    func=deprecated_formulas.calc_peak_pressure, return_keys=["peak_pressure"], name="calc_peak_pressure"
)
calc_magnetic_field_on_axis = Algorithm.from_single_function(
    lambda product_of_magnetic_field_and_radius, major_radius: product_of_magnetic_field_and_radius / major_radius,
    return_keys=["magnetic_field_on_axis"],
    name="calc_magnetic_field_on_axis",
)
require_P_rad_less_than_P_in = Algorithm.from_single_function(
    lambda P_in, P_radiation: np.minimum(P_radiation, P_in), return_keys=["P_radiation"], name="require_P_rad_less_than_P_in"
)
