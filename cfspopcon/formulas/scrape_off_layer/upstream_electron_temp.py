"""Routines to calculate the upstream electron temperature."""

from typing import Union

import xarray as xr

from ...unit_handling import Quantity


def calc_upstream_electron_temp(
    target_electron_temp: Union[Quantity, xr.DataArray],
    parallel_heat_flux_density: Union[Quantity, xr.DataArray],
    parallel_connection_length: Union[Quantity, xr.DataArray],
    kappa_e0: Union[Quantity, xr.DataArray],
    SOL_conduction_fraction: Union[float, xr.DataArray] = 1.0,
) -> Union[Quantity, xr.DataArray]:
    """Calculate the upstream electron temperature assuming Spitzer-Harm heat conductivity.

    Equation 38 from :cite:`stangeby_2018`, keeping the dependence on target_electron_temp.

    Args:
        target_electron_temp: [eV]
        parallel_heat_flux_density: [GW/m^2]
        parallel_connection_length: [m]
        kappa_e0: [W / (eV**3.5 m)]
        SOL_conduction_fraction: [eV]

    Returns:
        upstream_electron_temp [eV]
    """
    return (
        target_electron_temp**3.5 + 3.5 * (SOL_conduction_fraction * parallel_heat_flux_density * parallel_connection_length / kappa_e0)
    ) ** (2.0 / 7.0)
