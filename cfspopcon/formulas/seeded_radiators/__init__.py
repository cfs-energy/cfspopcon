"""Routines to calculate the concentration of a seeded impurity required to achieve a radiated power target."""

from .core_radiator_conc import calc_extrinsic_core_radiator
from .edge_radiator_conc import calc_edge_impurity_concentration

__all__ = [
    "calc_edge_impurity_concentration",
    "calc_extrinsic_core_radiator",
]
