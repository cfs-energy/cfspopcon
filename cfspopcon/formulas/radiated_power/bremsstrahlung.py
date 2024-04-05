"""Calculate the inherent Bremsstrahlung radiated power."""

import numpy as np
from numpy import float64
from numpy.typing import NDArray

from ...algorithm_class import Algorithm
from ...unit_handling import ureg, wraps_ufunc
from ..geometry.volume_integral import integrate_profile_over_volume


@Algorithm.register_algorithm(return_keys=["P_rad_bremsstrahlung"])
@wraps_ufunc(
    return_units=dict(P_rad_bremsstrahlung=ureg.MW),
    input_units=dict(
        rho=ureg.dimensionless,
        electron_density_profile=ureg.n19,
        electron_temp_profile=ureg.keV,
        z_effective=ureg.dimensionless,
        plasma_volume=ureg.m**3,
    ),
    input_core_dims=[("dim_rho",), ("dim_rho",), ("dim_rho",), (), ()],
)
def calc_bremsstrahlung_radiation(
    rho: NDArray[float64],
    electron_density_profile: NDArray[float64],
    electron_temp_profile: NDArray[float64],
    z_effective: float,
    plasma_volume: float,
) -> float:
    """Calculate the Bremsstrahlung radiated power due to the main plasma.

    Formula 13 in :cite:`stott_feasibility_2005`

    Args:
        electron_density_profile: [1e19 m^-3] :term:`glossary link<electron_density_profile>`
        electron_temp_profile: [keV] :term:`glossary link<electron_temp_profile>`
        z_effective: [~] :term:`glossary link<z_effective>`
        rho: [~] :term:`glossary link<rho>`
        plasma_volume: [m^3] :term:`glossary link<plasma_volume>`

    Returns:
        Radiated bremsstrahlung power per cubic meter [MW / m^3]
    """
    ne20 = electron_density_profile / 10

    Tm = 511.0  # keV, Tm = m_e * c**2
    xrel = (1.0 + 2.0 * electron_temp_profile / Tm) * (
        1.0 + (2.0 / z_effective) * (1.0 - 1.0 / (1.0 + electron_temp_profile / Tm))
    )  # relativistic correction factor

    fKb = ne20**2 * np.sqrt(electron_temp_profile) * xrel
    Kb = integrate_profile_over_volume.unitless_func(fKb, rho, plasma_volume)  # radial profile factor

    P_brem = 5.35e-3 * z_effective * Kb  # volume-averaged bremsstrahlung radiaton in MW

    return P_brem

Algorithm.from_single_function(
    func = lambda rho, electron_density_profile, electron_temp_profile, plasma_volume: calc_bremsstrahlung_radiation(rho, electron_density_profile, electron_temp_profile, 1.0, plasma_volume),
    name = "calc_P_rad_hydrogen_bremsstrahlung",
    return_keys=["P_rad_hydrogen_bremsstrahlung"]
)
