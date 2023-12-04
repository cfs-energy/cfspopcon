"""Routine to calculate the upstream SOL collisionality."""

from typing import Union

import xarray as xr

from ...unit_handling import Quantity, ureg, wraps_ufunc


@wraps_ufunc(
    return_units=dict(upstream_SOL_collisionality=ureg.dimensionless),
    input_units=dict(
        upstream_electron_density=ureg.m**-3, upstream_electron_temp=ureg.eV, parallel_connection_length=ureg.m
    ),
)
def calc_upstream_SOL_collisionality(
    upstream_electron_density: Union[Quantity, xr.DataArray],
    upstream_electron_temp: Union[Quantity, xr.DataArray],
    parallel_connection_length: Union[Quantity, xr.DataArray],
) -> Union[Quantity, xr.DataArray]:
    """Calculate the upstream SOL collisionality.

    Equation XX

    Args:
        upstream_electron_density: [m^-3]
        upstream_electron_temp: [eV]
        parallel_connection_length: [m]

    Returns:
        upstream_SOL_collisionality
    """
    return (1e-16*parallel_connection_length*upstream_electron_density/upstream_electron_temp**2)
