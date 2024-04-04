"""Calculate the average fuel mass in atomic mass units."""
from typing import Callable

from ...algorithm_class import Algorithm
from ...named_options import ReactionType
from ...unit_handling import ureg, wraps_ufunc

FUEL_MASS_AMU: dict[ReactionType, Callable[[float], float]] = {
    ReactionType.DT: lambda heavier_fuel_species_fraction: 2.0 + heavier_fuel_species_fraction,
    ReactionType.DD: lambda _: 2.0,
    ReactionType.DHe3: lambda heavier_fuel_species_fraction: 2.0 + heavier_fuel_species_fraction,
    ReactionType.pB11: lambda heavier_fuel_species_fraction: 11.0 * heavier_fuel_species_fraction
    + 1.0 * (1.0 - heavier_fuel_species_fraction),
}


@Algorithm.register_algorithm(return_keys=["fuel_average_mass_number"])
@wraps_ufunc(
    return_units=dict(fuel_average_mass_number=ureg.amu),
    input_units=dict(fusion_reaction=None, heavier_fuel_species_fraction=ureg.dimensionless),
)
def calc_average_fuel_ion_mass(fusion_reaction: ReactionType, heavier_fuel_species_fraction: float) -> float:
    """Calculate the average mass of the fuel ions, based on reaction type and fuel mixture ratio.

    Args:
        fusion_reaction: reaction type.
        heavier_fuel_species_fraction: n_heavier / (n_heavier + n_lighter) number fraction.

    Returns:
        :term:`fuel_average_mass_number` [amu]
    """
    return FUEL_MASS_AMU[fusion_reaction](heavier_fuel_species_fraction)
