"""Algorithms constructed by combining several smaller algorithms."""
from .algorithm_class import CompositeAlgorithm
from .beta import calc_beta
from .core_radiated_power import calc_core_radiated_power
from .extrinsic_core_radiator import calc_extrinsic_core_radiator
from .fusion_gain import calc_fusion_gain
from .geometry import calc_geometry
from .heat_exhaust import calc_heat_exhaust
from .ohmic_power import calc_ohmic_power
from .peaked_profiles import calc_peaked_profiles
from .power_balance_from_tau_e import calc_power_balance_from_tau_e
from .q_star_from_plasma_current import calc_q_star_from_plasma_current
from .single_functions import (
    calc_auxillary_power,
    calc_average_ion_temp,
    calc_average_total_pressure,
    calc_bootstrap_fraction,
    calc_confinement_transition_threshold_power,
    calc_current_relaxation_time,
    calc_f_rad_core,
    calc_fuel_average_mass_number,
    calc_greenwald_fraction,
    calc_normalised_collisionality,
    calc_P_SOL,
    calc_peak_pressure,
    calc_ratio_P_LH,
    calc_rho_star,
    calc_triple_product,
    require_P_rad_less_than_P_in,
)
from .two_point_model_fixed_tet import two_point_model_fixed_tet
from .zeff_and_dilution_from_impurities import calc_zeff_and_dilution_from_impurities

predictive_popcon = CompositeAlgorithm(
    [
        calc_geometry,
        calc_q_star_from_plasma_current,
        calc_fuel_average_mass_number,
        calc_average_ion_temp,
        calc_zeff_and_dilution_from_impurities,
        calc_power_balance_from_tau_e,
        calc_beta,
        calc_peaked_profiles,
        calc_core_radiated_power,
        require_P_rad_less_than_P_in,
        calc_extrinsic_core_radiator,
        calc_peaked_profiles,
        calc_fusion_gain,
        calc_bootstrap_fraction,
        calc_ohmic_power,
        calc_auxillary_power,
        calc_P_SOL,
        calc_average_total_pressure,
        calc_heat_exhaust,
        two_point_model_fixed_tet,
        calc_greenwald_fraction,
        calc_confinement_transition_threshold_power,
        calc_ratio_P_LH,
        calc_f_rad_core,
        calc_normalised_collisionality,
        calc_rho_star,
        calc_triple_product,
        calc_peak_pressure,
        calc_current_relaxation_time,
    ],
    name="predictive_popcon",
)
