"""Module to perform simple scrape-off-layer calculations.

These are mostly based on the two-point-model, from :cite:`stangeby_2018`.
"""
from .edge_impurity_concentration import build_L_int_integrator, calc_required_edge_impurity_concentration

__all__ = [
    "calc_required_edge_impurity_concentration",
    "build_L_int_integrator",
]
