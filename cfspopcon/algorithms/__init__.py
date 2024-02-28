"""POPCON algorithms."""
from typing import Union

from ..named_options import Algorithms
from .algorithm_class import Algorithm, CompositeAlgorithm
from .beta import calc_beta
from .composite_algorithm import predictive_popcon
from .core_radiated_power import calc_core_radiated_power
from .edge_impurity_concentration import calc_edge_impurity_concentration
from .extrinsic_core_radiator import calc_extrinsic_core_radiator
from .fusion_gain import calc_fusion_gain
from .geometry import calc_geometry
from .heat_exhaust import calc_heat_exhaust
from .ohmic_power import calc_ohmic_power
from .peaked_profiles import calc_peaked_profiles
from .plasma_current_from_q_star import calc_plasma_current_from_q_star
from .power_balance_from_tau_e import calc_power_balance_from_tau_e
from .q_star_from_plasma_current import calc_q_star_from_plasma_current
from .single_functions import SINGLE_FUNCTIONS
from .two_point_model_fixed_fpow import two_point_model_fixed_fpow
from .two_point_model_fixed_qpart import two_point_model_fixed_qpart
from .two_point_model_fixed_tet import two_point_model_fixed_tet
from .use_LOC_tau_e_below_threshold import use_LOC_tau_e_below_threshold
from .zeff_and_dilution_from_impurities import calc_zeff_and_dilution_from_impurities

ALGORITHMS: dict[Algorithms, Union[Algorithm, CompositeAlgorithm]] = {
    Algorithms["calc_beta"]: calc_beta,
    Algorithms["calc_core_radiated_power"]: calc_core_radiated_power,
    Algorithms["calc_extrinsic_core_radiator"]: calc_extrinsic_core_radiator,
    Algorithms["calc_fusion_gain"]: calc_fusion_gain,
    Algorithms["calc_geometry"]: calc_geometry,
    Algorithms["calc_heat_exhaust"]: calc_heat_exhaust,
    Algorithms["calc_ohmic_power"]: calc_ohmic_power,
    Algorithms["calc_peaked_profiles"]: calc_peaked_profiles,
    Algorithms["calc_plasma_current_from_q_star"]: calc_plasma_current_from_q_star,
    Algorithms["calc_power_balance_from_tau_e"]: calc_power_balance_from_tau_e,
    Algorithms["predictive_popcon"]: predictive_popcon,
    Algorithms["calc_q_star_from_plasma_current"]: calc_q_star_from_plasma_current,
    Algorithms["two_point_model_fixed_fpow"]: two_point_model_fixed_fpow,
    Algorithms["two_point_model_fixed_qpart"]: two_point_model_fixed_qpart,
    Algorithms["two_point_model_fixed_tet"]: two_point_model_fixed_tet,
    Algorithms["calc_zeff_and_dilution_from_impurities"]: calc_zeff_and_dilution_from_impurities,
    Algorithms["use_LOC_tau_e_below_threshold"]: use_LOC_tau_e_below_threshold,
    Algorithms["calc_edge_impurity_concentration"]: calc_edge_impurity_concentration,
    **SINGLE_FUNCTIONS,
}


def get_algorithm(algorithm: Union[Algorithms, str]) -> Union[Algorithm, CompositeAlgorithm]:
    """Accessor for algorithms."""
    if isinstance(algorithm, str):
        algorithm = Algorithms[algorithm]

    return ALGORITHMS[algorithm]


__all__ = [
    "ALGORITHMS",
    "get_algorithm",
]

for key in ALGORITHMS.keys():
    __all__.append(key.name)
