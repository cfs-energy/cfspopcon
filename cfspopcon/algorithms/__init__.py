"""POPCON algorithms."""
from .beta import calc_beta
from .core_radiated_power import calc_core_radiated_power
from .edge_impurity_concentration import calc_edge_impurity_concentration
from .extrinsic_core_radiator import calc_extrinsic_core_radiator
from .fusion_gain import calc_fusion_gain
from .heat_exhaust import calc_heat_exhaust
from .ohmic_power import calc_ohmic_power
from .peaked_profiles import calc_peaked_profiles
from .power_balance_from_tau_e import calc_power_balance_from_tau_e
from .single_functions import (
    calc_auxillary_power,
    calc_average_ion_temp,
    calc_average_total_pressure,
    calc_bootstrap_fraction,
    calc_confinement_transition_threshold_power,
    calc_current_relaxation_time,
    calc_f_rad_core,
    calc_greenwald_fraction,
    calc_magnetic_field_on_axis,
    calc_normalised_collisionality,
    calc_P_SOL,
    calc_peak_pressure,
    calc_plasma_stored_energy,
    calc_ratio_P_LH,
    calc_rho_star,
    calc_triple_product,
    calc_upstream_electron_density,
    require_P_rad_less_than_P_in,
)
from .two_point_model_fixed_fpow import two_point_model_fixed_fpow
from .two_point_model_fixed_qpart import two_point_model_fixed_qpart
from .two_point_model_fixed_tet import two_point_model_fixed_tet
from .use_LOC_tau_e_below_threshold import use_LOC_tau_e_below_threshold
from .zeff_and_dilution_from_impurities import calc_zeff_and_dilution_from_impurities

__all__ = [
    "get_algorithm",
    "calc_beta",
    "calc_core_radiated_power",
    "calc_edge_impurity_concentration",
    "calc_extrinsic_core_radiator",
    "calc_fusion_gain",
    "calc_heat_exhaust",
    "calc_ohmic_power",
    "calc_peaked_profiles",
    "calc_power_balance_from_tau_e",
    "two_point_model_fixed_fpow",
    "two_point_model_fixed_qpart",
    "two_point_model_fixed_tet",
    "use_LOC_tau_e_below_threshold",
    "calc_zeff_and_dilution_from_impurities",
    "calc_confinement_transition_threshold_power",
    "calc_ratio_P_LH",
    "calc_f_rad_core",
    "calc_normalised_collisionality",
    "calc_rho_star",
    "calc_triple_product",
    "calc_greenwald_fraction",
    "calc_current_relaxation_time",
    "calc_peak_pressure",
    "calc_average_total_pressure",
    "calc_bootstrap_fraction",
    "calc_auxillary_power",
    "calc_average_ion_temp",
    "calc_fuel_average_mass_number",
    "calc_magnetic_field_on_axis",
    "require_P_rad_less_than_P_in",
    "calc_P_SOL",
    "calc_plasma_stored_energy",
    "calc_upstream_electron_density",
]
