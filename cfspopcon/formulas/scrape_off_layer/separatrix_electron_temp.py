"""Routines to calculate the upstream electron temperature."""

from typing import Union

import xarray as xr

from ...algorithm_class import Algorithm
from ...unit_handling import Unitfull


@Algorithm.register_algorithm(return_keys=["separatrix_electron_temp"])
def calc_separatrix_electron_temp(
    target_electron_temp: Unitfull,
    q_parallel: Unitfull,
    parallel_connection_length: Unitfull,
    kappa_e0: Unitfull,
    SOL_conduction_fraction: Union[float, xr.DataArray] = 1.0,
) -> Unitfull:
    """Calculate the upstream electron temperature assuming Spitzer-Harm heat conductivity.

    Equation 38 from :cite:`stangeby_2018`, keeping the dependence on target_electron_temp.

    Args:
        target_electron_temp: [eV] :term:`glossary link<target_electron_temp>`
        q_parallel: [GW/m^2] :term:`glossary link<q_parallel>`
        parallel_connection_length: [m] :term:`glossary link<parallel_connection_length>`
        kappa_e0: [W / (eV**3.5 m)] :term:`glossary link<kappa_e0>`
        SOL_conduction_fraction: [eV] :term:`glossary link<SOL_conduction_fraction>`

    Returns:
        separatrix_electron_temp [eV]
    """
    return (target_electron_temp**3.5 + 3.5 * (SOL_conduction_fraction * q_parallel * parallel_connection_length / kappa_e0)) ** (2.0 / 7.0)
