"""The physics formulas behind the POPCON analysis, one subpackage per topic.

This package is the bundled plugin of cfspopcon: the first use of the registry registers every
algorithm defined here.
"""

from . import (
    atomic_data,
    auxiliary_power,
    energy_confinement,
    fusion_power,
    geometry,
    impurities,
    metrics,
    plasma_current,
    plasma_pressure,
    plasma_profiles,
    radiated_power,
    scrape_off_layer,
    separatrix_conditions,
)

__all__ = [
    "atomic_data",
    "auxiliary_power",
    "energy_confinement",
    "fusion_power",
    "geometry",
    "impurities",
    "metrics",
    "plasma_current",
    "plasma_pressure",
    "plasma_profiles",
    "radiated_power",
    "scrape_off_layer",
    "separatrix_conditions",
]
