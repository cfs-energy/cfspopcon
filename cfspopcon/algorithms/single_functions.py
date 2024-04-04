"""Algorithm wrappers for single functions which don't fit into larger algorithms."""
import numpy as np

from .. import deprecated_formulas
from ..algorithm_class import Algorithm
from ..unit_handling import ureg

calc_confinement_transition_threshold_power = Algorithm.from_single_function(
    deprecated_formulas.calc_LH_transition_threshold_power,
    return_keys=["P_LH_thresh"],
    name="calc_confinement_transition_threshold_power",
)
calc_ratio_P_LH = Algorithm.from_single_function(
    func=lambda P_sol, P_LH_thresh: P_sol / P_LH_thresh, return_keys=["ratio_of_P_SOL_to_P_LH"], name="calc_ratio_P_LH"
)
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
calc_greenwald_fraction = Algorithm.from_single_function(
    func=deprecated_formulas.calc_greenwald_fraction, return_keys=["greenwald_fraction"], name="calc_greenwald_fraction"
)
calc_current_relaxation_time = Algorithm.from_single_function(
    func=deprecated_formulas.calc_current_relaxation_time, return_keys=["current_relaxation_time"], name="calc_current_relaxation_time"
)
calc_peak_pressure = Algorithm.from_single_function(
    func=deprecated_formulas.calc_peak_pressure, return_keys=["peak_pressure"], name="calc_peak_pressure"
)
calc_average_total_pressure = Algorithm.from_single_function(
    lambda average_electron_density, average_electron_temp, average_ion_temp: average_electron_density
    * (average_electron_temp + average_ion_temp),
    return_keys=["average_total_pressure"],
    name="calc_average_total_pressure",
)
calc_bootstrap_fraction = Algorithm.from_single_function(
    deprecated_formulas.calc_bootstrap_fraction, return_keys=["bootstrap_fraction"], name="calc_bootstrap_fraction"
)
calc_auxillary_power = Algorithm.from_single_function(
    lambda P_external, P_ohmic: (P_external - P_ohmic).clip(min=0.0 * ureg.MW), return_keys=["P_auxillary"], name="calc_auxillary_power"
)

calc_magnetic_field_on_axis = Algorithm.from_single_function(
    lambda product_of_magnetic_field_and_radius, major_radius: product_of_magnetic_field_and_radius / major_radius,
    return_keys=["magnetic_field_on_axis"],
    name="calc_magnetic_field_on_axis",
)
require_P_rad_less_than_P_in = Algorithm.from_single_function(
    lambda P_in, P_radiation: np.minimum(P_radiation, P_in), return_keys=["P_radiation"], name="require_P_rad_less_than_P_in"
)
calc_P_SOL = Algorithm.from_single_function(
    lambda P_in, P_radiation: np.maximum(P_in - P_radiation, 0.0), return_keys=["P_sol"], name="calc_P_SOL"
)
calc_upstream_electron_density = Algorithm.from_single_function(
    lambda nesep_over_nebar, average_electron_density: nesep_over_nebar * average_electron_density,
    return_keys=["upstream_electron_density"],
    name="calc_upstream_electron_density",
)
