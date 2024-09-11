"""Calculate the mean charge state of an impurity for given plasma conditions."""

import numpy as np
import xarray as xr

from ...algorithm_class import Algorithm
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
    if isinstance(atomic_data, xr.DataArray):
        atomic_data = atomic_data.item()

    return _calc_impurity_charge_state(average_electron_density, average_electron_temp, impurity_concentration.dim_species, atomic_data)


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
    average_electron_temp, average_electron_density = atomic_data.nearest_neighbour_off_grid(  # type:ignore[assignment]
        impurity_species,
        average_electron_temp,
        average_electron_density,
    )
    interpolator = atomic_data.coronal_Z_interpolators[impurity_species]
    interpolated_values = np.power(10, interpolator((np.log10(average_electron_temp), np.log10(average_electron_density))))

    atomic_number = {
        AtomicSpecies.Hydrogen: 1,
        AtomicSpecies.Deuterium: 1,
        AtomicSpecies.Tritium: 1,
        AtomicSpecies.Helium: 2,
        AtomicSpecies.Lithium: 3,
        AtomicSpecies.Beryllium: 4,
        AtomicSpecies.Boron: 5,
        AtomicSpecies.Carbon: 6,
        AtomicSpecies.Nitrogen: 7,
        AtomicSpecies.Oxygen: 8,
        AtomicSpecies.Neon: 10,
        AtomicSpecies.Argon: 18,
        AtomicSpecies.Krypton: 36,
        AtomicSpecies.Xenon: 54,
        AtomicSpecies.Tungsten: 74,
    }

    interpolated_values = np.minimum(interpolated_values, atomic_number[impurity_species])
    interpolated_values = np.maximum(interpolated_values, 0)
    return interpolated_values  # type:ignore[no-any-return]
