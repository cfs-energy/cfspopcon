"""The physics formulas behind the POPCON analysis, one subpackage per topic.

This package is the bundled plugin of cfspopcon. You can register it deliberately, like any other
plugin, with ``register_plugin("cfspopcon.formulas")``.

If you did not register the bundled plugin, the first use of the registry will trigger it
automatically::

    import cfspopcon

    # registers the bundled plugin
    volume_algorithm = cfspopcon.registry["calc_plasma_volume"]
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
