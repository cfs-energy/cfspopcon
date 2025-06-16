"""Calculate the mean charge state of an impurity for given plasma conditions."""

import numpy as np
import xarray as xr

from ...algorithm_class import Algorithm
from ...helpers import get_item
from ...named_options import AtomicSpecies
from ...unit_handling import Unitfull, ureg, wraps_ufunc
from ..atomic_data import AtomicData


@Algorithm.register_algorithm(return_keys=["impurity_charge_state"])
def calc_impurity_charge_state(
    average_electron_density: Unitfull,
    average_electron_temp: Unitfull,
    impurity_concentration: xr.DataArray,
    atomic_data: xr.DataArray,
) -> Unitfull:
    """Calculate the impurity charge state for each species in impurity_concentration.

    Args:
        average_electron_density: :term:`glossary link<average_electron_density>`
        average_electron_temp: :term:`glossary link<average_electron_temp>`
        impurity_concentration: :term:`glossary link<impurity_concentration>`
        atomic_data: :term:`glossary link<atomic_data>`

    Returns:
        :term:`impurity_charge_state`
    """
    return _calc_impurity_charge_state(
        average_electron_density, average_electron_temp, impurity_concentration.dim_species, get_item(atomic_data)
    )


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
def _calc_impurity_charge_state(
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
    interpolator = atomic_data.get_coronal_Z_interpolator(impurity_species)
    interpolated_values = interpolator(electron_density=average_electron_density, electron_temp=average_electron_temp, allow_extrap=True)

    atomic_number = atomic_data.datasets[impurity_species].atomic_number

    interpolated_values = np.minimum(interpolated_values, atomic_number)
    interpolated_values = np.maximum(interpolated_values, 0.0)
    return interpolated_values  # type:ignore[no-any-return]
