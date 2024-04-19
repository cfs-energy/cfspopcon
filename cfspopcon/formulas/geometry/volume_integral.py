"""Common functionality shared between other functions."""
import numpy as np
from numpy import float64
from numpy.typing import NDArray

from ...unit_handling import ureg, wraps_ufunc


@wraps_ufunc(
    input_units=dict(array_per_m3=ureg.m**-3, rho=ureg.dimensionless, plasma_volume=ureg.m**3),
    return_units=dict(volume_integrated_value=ureg.dimensionless),
    input_core_dims=[("dim_rho",), ("dim_rho",), ()],
)
def integrate_profile_over_volume(
    array_per_m3: NDArray[float64],
    rho: NDArray[float64],
    plasma_volume: float,
) -> float:
    """Approximate the volume integral of a profile given as a function of rho.

    Args:
        array_per_m3: a profile of values [units * m^-3]
        rho: [~] :term:`glossary link<rho>`
        plasma_volume: [m^3] :term:`glossary link<plasma_volume>`

    Returns:
         volume_integrated_value [units]
    """
    drho = rho[1] - rho[0]
    result: float = np.sum(array_per_m3 * 2.0 * rho * drho) * plasma_volume
    return result
