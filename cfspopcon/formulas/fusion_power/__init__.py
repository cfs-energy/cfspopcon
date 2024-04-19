"""Routines to calculate the fusion power and gain."""

from . import fusion_data
from .average_fuel_ion_mass import calc_average_fuel_ion_mass
from .fusion_data import (
    DDFusionBoschHale,
    DDFusionHively,
    DHe3Fusion,
    DTFusionBoschHale,
    DTFusionHively,
    FusionReaction,
    pB11Fusion,
)
from .fusion_gain import calc_fusion_gain, calc_triple_product
from .fusion_rates import calc_fusion_power, calc_neutron_flux_to_walls

__all__ = [
    "fusion_data",
    "calc_average_fuel_ion_mass",
    "FusionReaction",
    "DTFusionBoschHale",
    "DTFusionHively",
    "DDFusionBoschHale",
    "DDFusionHively",
    "DHe3Fusion",
    "pB11Fusion",
    "calc_fusion_gain",
    "calc_triple_product",
    "calc_fusion_power",
    "calc_neutron_flux_to_walls",
]
