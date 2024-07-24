"""Calculate the average fuel mass in atomic mass units."""

import xarray as xr

from ...algorithm_class import Algorithm
from ...unit_handling import Unitfull
from .fusion_data import (
    REACTIONS,
    DDFusionBoschHale,
    DDFusionHively,
    DHe3Fusion,
    DTFusionBoschHale,
    DTFusionHively,
    pB11Fusion,
)


@Algorithm.register_algorithm(return_keys=["fuel_average_mass_number"])
def calc_average_fuel_ion_mass(
    fusion_reaction: str, heavier_fuel_species_fraction: Unitfull
) -> Unitfull:
    """Calculate the average mass of the fuel ions, based on reaction type and fuel mixture ratio.

    Args:
        fusion_reaction: reaction type.
        heavier_fuel_species_fraction: n_heavier / (n_heavier + n_lighter) number fraction.

    Returns:
        :term:`fuel_average_mass_number` [amu]
    """
    if isinstance(fusion_reaction, xr.DataArray):
        fusion_reaction = fusion_reaction.item()
    reaction = REACTIONS[fusion_reaction]
    if isinstance(
        reaction, (DTFusionBoschHale, DTFusionHively, DHe3Fusion, pB11Fusion)
    ):
        return reaction.calc_average_fuel_ion_mass(heavier_fuel_species_fraction)
    elif isinstance(reaction, (DDFusionBoschHale, DDFusionHively)):
        return reaction.calc_average_fuel_ion_mass()
    else:
        raise NotImplementedError(
            f"No implementation for calc_average_fuel_ion_mass for {fusion_reaction}"
        )
