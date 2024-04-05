"""POPCON algorithms."""
from .core_radiated_power import calc_core_radiated_power
from .edge_impurity_concentration import calc_edge_impurity_concentration
from .extrinsic_core_radiator import calc_extrinsic_core_radiator
from .single_functions import (
    calc_f_rad_core,
    require_P_rad_less_than_P_in,
)

__all__ = [
    "calc_core_radiated_power",
    "calc_edge_impurity_concentration",
    "calc_extrinsic_core_radiator",
    "calc_ohmic_power",
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
    "calc_fuel_average_mass_number",
    "calc_magnetic_field_on_axis",
    "require_P_rad_less_than_P_in",
    "calc_P_SOL",
]
