"""Calculate the effect of impurities on the effective charge and dilution."""
import numpy as np
import xarray as xr

from ..named_options import AtomicSpecies
from ..read_atomic_data import AtomicData
from ..unit_handling import ureg, wraps_ufunc


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
    return float(
        atomic_data.eval_interpolator(
            electron_density=np.atleast_1d(average_electron_density),
            electron_temp=np.atleast_1d(average_electron_temp),
            kind=AtomicData.CoronalZ,
            species=impurity_species,
            allow_extrapolation=True,
        )
    )


def calc_change_in_zeff(impurity_charge_state: float, impurity_concentration: xr.DataArray) -> xr.DataArray:
    """Calculate the change in the effective charge due to the specified impurities.

    Args:
        impurity_charge_state: [~] :term:`glossary link<impurity_charge_state>`
        impurity_concentration: [~] :term:`glossary link<impurity_concentration>`

    Returns:
        change in zeff [~]
    """
    return impurity_charge_state * (impurity_charge_state - 1.0) * impurity_concentration


def calc_change_in_dilution(impurity_charge_state: float, impurity_concentration: xr.DataArray) -> xr.DataArray:
    """Calculate the change in n_fuel/n_e due to the specified impurities.

    Args:
        impurity_charge_state: [~] :term:`glossary link<impurity_charge_state>`
        impurity_concentration: [~] :term:`glossary link<impurity_concentration>`

    Returns:
        change in dilution [~]
    """
    return impurity_charge_state * impurity_concentration
