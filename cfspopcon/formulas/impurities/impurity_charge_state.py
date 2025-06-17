"""Calculate the mean charge state of an impurity for given plasma conditions."""

import numpy as np
import xarray as xr

from ...algorithm_class import Algorithm
from ...helpers import get_item
from ...unit_handling import Unitfull
from ..atomic_data import AtomicData


@Algorithm.register_algorithm(return_keys=["impurity_charge_state"])
def calc_impurity_charge_state(
    average_electron_density: Unitfull,
    average_electron_temp: Unitfull,
    impurity_concentration: xr.DataArray,
    atomic_data: AtomicData | xr.DataArray,
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
    atomic_data = get_item(atomic_data)

    def calc_mean_z(impurity_concentration: xr.DataArray) -> xr.DataArray:
        species = get_item(impurity_concentration.dim_species)
        interpolator = atomic_data.get_coronal_Z_interpolator(species)
        mean_z = interpolator.eval(electron_density=average_electron_density, electron_temp=average_electron_temp, allow_extrap=True)

        mean_z = np.minimum(mean_z, atomic_data.datasets[species].atomic_number)
        mean_z = np.maximum(mean_z, 0.0)
        return mean_z  # type:ignore[no-any-return]

    return impurity_concentration.groupby("dim_species").map(calc_mean_z)
