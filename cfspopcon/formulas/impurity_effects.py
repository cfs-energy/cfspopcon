"""Calculate the effect of impurities on the effective charge and dilution."""
import numpy as np
import xarray as xr

from ..algorithm_class import Algorithm
from ..named_options import AtomicSpecies
from ..read_atomic_data import AtomicData
from ..unit_handling import ureg, wraps_ufunc


@Algorithm.register_algorithm(return_keys=["impurity_charge_state"])
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
        impurity_species, average_electron_temp, average_electron_density  # type:ignore[arg-type]
    )
    interpolator = atomic_data.coronal_Z_interpolators[impurity_species]
    interpolated_values = np.power(10, interpolator((np.log10(average_electron_temp), np.log10(average_electron_density))))

    interpolated_values = np.minimum(interpolated_values, impurity_species.value)
    interpolated_values = np.maximum(interpolated_values, 0)
    return interpolated_values  # type:ignore[no-any-return]


@Algorithm.register_algorithm(return_keys=["change_in_zeff"])
def calc_change_in_zeff(impurity_charge_state: float, impurity_concentration: xr.DataArray) -> xr.DataArray:
    """Calculate the change in the effective charge due to the specified impurities.

    Args:
        impurity_charge_state: [~] :term:`glossary link<impurity_charge_state>`
        impurity_concentration: [~] :term:`glossary link<impurity_concentration>`

    Returns:
        change_in_zeff [~]
    """
    return impurity_charge_state * (impurity_charge_state - 1.0) * impurity_concentration


@Algorithm.register_algorithm(return_keys=["change_in_dilution"])
def calc_change_in_dilution(impurity_charge_state: float, impurity_concentration: xr.DataArray) -> xr.DataArray:
    """Calculate the change in n_fuel/n_e due to the specified impurities.

    Args:
        impurity_charge_state: [~] :term:`glossary link<impurity_charge_state>`
        impurity_concentration: [~] :term:`glossary link<impurity_concentration>`

    Returns:
        change_in_dilution [~]
    """
    return impurity_charge_state * impurity_concentration
