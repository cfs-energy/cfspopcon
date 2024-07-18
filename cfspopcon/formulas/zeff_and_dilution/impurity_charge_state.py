"""Calculate the mean charge state of an impurity for given plasma conditions."""

import numpy as np

from ...named_options import AtomicSpecies
from ...unit_handling import ureg, wraps_ufunc
from ..read_atomic_data import AtomicData


@wraps_ufunc(
    return_units=dict(mean_charge_state=ureg.dimensionless),
    input_units=dict(
        average_electron_density=ureg.m**-3,
        average_electron_temp=ureg.eV,
        impurity_species=None,
        atomic_data=None,
    ),
    pass_as_kwargs=("atomic_data",),
)
def calc_impurity_charge_state(
    average_electron_density: float,
    average_electron_temp: float,
    impurity_species: AtomicSpecies,
    atomic_data: AtomicData,
) -> float:
    """Calculate the impurity charge state of the specified impurity species.

    Args:
        average_electron_density: [m^-3] :term:`glossary link<average_electron_density>`
        average_electron_temp: [eV] :term:`glossary link<average_electron_temp>`
        impurity_species: [] :term:`glossary link<impurity_species>`
        atomic_data: :term:`glossary link<atomic_data>`

    Returns:
        :term:`impurity_charge_state`
    """
    average_electron_temp, average_electron_density = atomic_data.nearest_neighbour_off_grid(  # type:ignore[assignment]
        impurity_species,
        average_electron_temp,
        average_electron_density,  # type:ignore[arg-type]
    )
    interpolator = atomic_data.coronal_Z_interpolators[impurity_species]
    interpolated_values = np.power(10, interpolator((np.log10(average_electron_temp), np.log10(average_electron_density))))

    interpolated_values = np.minimum(interpolated_values, impurity_species.value)
    interpolated_values = np.maximum(interpolated_values, 0)
    return interpolated_values  # type:ignore[no-any-return]
