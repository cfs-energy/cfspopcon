"""Calculate the inherent (Bremsstrahlung and Synchrotron) radiated power."""

import numpy as np
from numpy import float64
from numpy.typing import NDArray

from ...unit_handling import ureg, wraps_ufunc
from ..helpers import integrate_profile_over_volume


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
    Kb = integrate_profile_over_volume(fKb, rho, plasma_volume)  # radial profile factor

    P_brem = 5.35e-3 * z_effective * Kb  # volume-averaged bremsstrahlung radiaton in MW

    return P_brem


@wraps_ufunc(
    return_units=dict(P_rad_synchrotron=ureg.MW),
    input_units=dict(
        rho=ureg.dimensionless,
        electron_density_profile=ureg.n19,
        electron_temp_profile=ureg.keV,
        major_radius=ureg.m,
        minor_radius=ureg.m,
        magnetic_field_on_axis=ureg.T,
        separatrix_elongation=ureg.dimensionless,
        plasma_volume=ureg.m**3,
    ),
    input_core_dims=[("dim_rho",), ("dim_rho",), ("dim_rho",), (), (), (), (), ()],
)
def calc_synchrotron_radiation(
    rho: NDArray[float64],
    electron_density_profile: NDArray[float64],
    electron_temp_profile: NDArray[float64],
    major_radius: float,
    minor_radius: float,
    magnetic_field_on_axis: float,
    separatrix_elongation: float,
    plasma_volume: float,
) -> float:
    """Calculate the Synchrotron radiated power due to the main plasma.

    This can be an important loss mechanism in high temperature plasmas.

    Formula 15 in :cite:`stott_feasibility_2005`

    For now this calculation assumes 90% wall reflectivity, consistent with stott_feasibility_2005.

    This calculation also assumes profiles of the form n(r) = n[1 - (r/a)**2]**alpha_n and
    T(r) = Tedge + (T - Tedge)[1 - (r/a)**gamma_T]**alpha_T. For now, these are assumed as
    gamma_T = 2, alpha_n = 0.5 and alpha_T = 1, consistent with stott_feasibility_2005.

    An alternative approach could be developed using formula 6 in :cite:`zohm_use_2019`, which assumes 80% wall reflectivity.

    Args:
        electron_density_profile: [1e19 m^-3] :term:`glossary link<electron_density_profile>`
        electron_temp_profile: [keV] :term:`glossary link<electron_temp_profile>`
        major_radius: [m] :term:`glossary link<major_radius>`
        minor_radius: [m] :term:`glossary link<minor_radius>`
        magnetic_field_on_axis: [T] :term:`glossary link<magnetic_field_on_axis>`
        separatrix_elongation: [~] :term:`glossary link<separatrix_elongation>`
        rho: [~] :term:`glossary link<rho>`
        plasma_volume: [m^3] :term:`glossary link<plasma_volume>`

    Returns:
        Radiated bremsstrahlung power per cubic meter [MW / m^3]
    """
    ne20 = electron_density_profile / 10

    Rw = 0.8  # wall reflectivity
    gamma_T = 2  # temperature profile inner exponent (2 is ~parabolic)
    alpha_n = 0.5  # density profile outer exponent (0.5 is rather broad)
    alpha_T = 1  # temperature profile outer exponent (1 is ~parabolic)

    # effective optical thickness
    rhoa = 6.04e3 * minor_radius * ne20 / magnetic_field_on_axis
    # profile peaking correction
    Ks = (
        (alpha_n + 3.87 * alpha_T + 1.46) ** (-0.79)
        * (1.98 + alpha_n) ** (1.36)
        * gamma_T**2.14
        * (gamma_T**1.53 + 1.87 * alpha_T - 0.16) ** (-1.33)
    )
    # aspect ratio correction
    Gs = 0.93 * (1 + 0.85 * np.exp(-0.82 * major_radius / minor_radius))

    # dimensionless parameter to account for plasma transparency and wall reflections
    Phi = (
        6.86e-5
        * separatrix_elongation ** (-0.21)
        * (16 + electron_temp_profile) ** (2.61)
        * ((rhoa / (1 - Rw)) ** (0.41) + 0.12 * electron_temp_profile) ** (-1.51)
        * Ks
        * Gs
    )

    P_sync_r = 6.25e-3 * ne20 * electron_temp_profile * magnetic_field_on_axis**2 * Phi
    P_sync = integrate_profile_over_volume(P_sync_r, rho, plasma_volume)

    return P_sync
