"""Formulas for POPCONs analysis."""

from . import energy_confinement_time_scalings, fusion_reaction_data, plasma_profile_data, radiated_power, separatrix_operational_space
from .average_fuel_ion_mass import calc_fuel_average_mass_number
from .beta import calc_beta_normalised, calc_beta_poloidal, calc_beta_toroidal, calc_beta_total
from .confinement_regime_threshold_powers import (
    calc_confinement_transition_threshold_power,
    calc_LI_transition_threshold_power,
    calLH_transition_condition_transition_threshold_power,
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
from .energy_confinement_time_scalings import calc_tau_e_and_P_in_from_scaling
from .figures_of_merit import calc_normalised_collisionality, calc_peak_pressure, calc_rho_star, calc_triple_product
from .fusion_rates import calc_fusion_power, calc_neutron_flux_to_walls
from .geometry import calc_plasma_surface_area, calc_plasma_volume
from .impurity_effects import calc_change_in_dilution, calc_change_in_zeff, calc_impurity_charge_state
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
    "separatrix_operational_space",
    "energy_confinement_time_scalings",
    "fusion_reaction_data",
    "calc_density_peaking",
    "calc_effective_collisionality",
    "plasma_profile_data",
    "calc_fuel_average_mass_number",
    "calc_B_pol_omp",
    "calc_B_tor_omp",
    "calc_beta_normalised",
    "calc_beta_poloidal",
    "calc_beta_toroidal",
    "calc_troyon_limit",
    "calc_greenwald_density_limit",
    "calc_beta_total",
    "calc_bootstrap_fraction",
    "calc_current_relaxation_time",
    "calc_f_shaping",
    "calc_fusion_power",
    "calc_greenwald_fraction",
    "calLH_transition_condition_transition_threshold_power",
    "calc_LI_transition_threshold_power",
    "calc_loop_voltage",
    "calc_resistivity_trapped_enhancement",
    "calc_neoclassical_loop_resistivity",
    "calc_neutron_flux_to_walls",
    "calc_normalised_collisionality",
    "calc_1D_plasma_profiles",
    "calc_ohmic_power",
    "calc_peak_pressure",
    "calc_plasma_current",
    "calc_plasma_surface_area",
    "calc_plasma_volume",
    "calc_rho_star",
    "calc_Spitzer_loop_resistivity",
    "calc_q_star",
    "calc_triple_product",
    "calc_tau_e_and_P_in_from_scaling",
    "thermal_calc_gain_factor",
    "calc_confinement_transition_threshold_power",
    "calc_change_in_zeff",
    "calc_change_in_dilution",
    "calc_bremsstrahlung_radiation",
    "calc_synchrotron_radiation",
    "calc_impurity_radiated_power_post_and_jensen",
    "calc_impurity_radiated_power_radas",
    "calc_impurity_radiated_power",
    "calc_impurity_radiated_power_mavrin_coronal",
    "calc_impurity_radiated_power_mavrin_noncoronal",
    "radiated_power",
    "calc_impurity_charge_state",
]
