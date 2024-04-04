"""Formulas for POPCONs analysis."""

from . import fusion_reaction_data, plasma_profile_data, radiated_power, scrape_off_layer_model
from .beta import calc_beta_normalised, calc_beta_poloidal, calc_beta_toroidal, calc_beta_total
from .confinement_regime_threshold_powers import (
    calc_LH_transition_threshold_power,
    calc_LI_transition_threshold_power,
)
from .current_drive import (
    calc_bootstrap_fraction,
    calc_current_relaxation_time,
    calc_f_shaping,
    calc_loop_voltage,
    calc_neoclassical_loop_resistivity,
    calc_ohmic_power,
    calc_plasma_current,
    calc_q_star,
    calc_resistivity_trapped_enhancement,
    calc_Spitzer_loop_resistivity,
)
from .density_peaking import calc_density_peaking, calc_effective_collisionality
from .divertor_metrics import calc_B_pol_omp, calc_B_tor_omp
from .figures_of_merit import calc_normalised_collisionality, calc_peak_pressure, calc_rho_star, calc_triple_product
from .fusion_rates import calc_fusion_power, calc_neutron_flux_to_walls
from .operational_limits import calc_greenwald_density_limit, calc_greenwald_fraction, calc_troyon_limit
from .plasma_profiles import calc_1D_plasma_profiles
from .Q_thermal_gain_factor import thermal_calc_gain_factor
from .radiated_power import (
    calc_bremsstrahlung_radiation,
    calc_impurity_radiated_power,
    calc_impurity_radiated_power_mavrin_coronal,
    calc_impurity_radiated_power_mavrin_noncoronal,
    calc_impurity_radiated_power_post_and_jensen,
    calc_impurity_radiated_power_radas,
    calc_synchrotron_radiation,
)

__all__ = [
    "calc_1D_plasma_profiles",
    "calc_B_pol_omp",
    "calc_B_tor_omp",
    "calc_beta_normalised",
    "calc_beta_poloidal",
    "calc_beta_toroidal",
    "calc_beta_total",
    "calc_bootstrap_fraction",
    "calc_bremsstrahlung_radiation",
    "calc_current_relaxation_time",
    "calc_density_peaking",
    "calc_effective_collisionality",
    "calc_f_shaping",
    "calc_fusion_power",
    "calc_greenwald_density_limit",
    "calc_greenwald_fraction",
    "calc_impurity_radiated_power_mavrin_coronal",
    "calc_impurity_radiated_power_mavrin_noncoronal",
    "calc_impurity_radiated_power_post_and_jensen",
    "calc_impurity_radiated_power_radas",
    "calc_impurity_radiated_power",
    "calc_LH_transition_threshold_power",
    "calc_LI_transition_threshold_power",
    "calc_loop_voltage",
    "calc_neoclassical_loop_resistivity",
    "calc_neutron_flux_to_walls",
    "calc_normalised_collisionality",
    "calc_ohmic_power",
    "calc_peak_pressure",
    "calc_plasma_current",
    "calc_q_star",
    "calc_resistivity_trapped_enhancement",
    "calc_rho_star",
    "calc_Spitzer_loop_resistivity",
    "calc_synchrotron_radiation",
    "calc_triple_product",
    "calc_troyon_limit",
    "fusion_reaction_data",
    "plasma_profile_data",
    "radiated_power",
    "scrape_off_layer_model",
    "thermal_calc_gain_factor",
]
