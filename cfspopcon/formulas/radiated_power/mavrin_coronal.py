"""Calculate the radiated power due to impurities, according to an analytical fitted curve from Mavrin 2018."""

import warnings

import numpy as np
from numpy import float64
from numpy.polynomial.polynomial import polyval
from numpy.typing import NDArray

from ...named_options import Impurity
from ...unit_handling import ureg, wraps_ufunc
from ..helpers import integrate_profile_over_volume


@wraps_ufunc(
    return_units=dict(radiated_power=ureg.MW),
    input_units=dict(
        rho=ureg.dimensionless,
        electron_temp_profile=ureg.keV,
        electron_density_profile=ureg.n19,
        impurity_concentration=ureg.dimensionless,
        impurity_species=None,
        plasma_volume=ureg.m**3,
    ),
    input_core_dims=[("dim_rho",), ("dim_rho",), ("dim_rho",), (), (), ()],
)
def calc_impurity_radiated_power_mavrin_coronal(  # noqa: PLR0912, PLR0915
    rho: NDArray[float64],
    electron_temp_profile: NDArray[float64],
    electron_density_profile: NDArray[float64],
    impurity_concentration: float,
    impurity_species: Impurity,
    plasma_volume: float,
) -> float:
    """Calculation of radiated power, using fits from A.A. Mavrin's 2018 paper.

    "Improved fits of coronal radiative cooling rates for high-temperature plasmas."

    :cite:`mavrin_improved_2018`

    Args:
        rho: [~] :term:`glossary link<rho>`
        electron_temp_profile: [keV] :term:`glossary link<electron_temp_profile>`
        electron_density_profile: [1e19 m^-3] :term:`glossary link<electron_density_profile>`
        impurity_species: [] :term:`glossary link<impurity_species>`
        impurity_concentration: [~] :term:`glossary link<impurity_concentration>`
        plasma_volume: [m^3] :term:`glossary link<plasma_volume>`

    Returns:
      k  [MW] Estimated radiation power due to this impurity
    """
    impurity_Z = impurity_species.value

    zimp = np.array([2, 3, 4, 6, 7, 8, 10, 18, 36, 54, 74])

    if impurity_Z not in zimp:  # pragma: no cover
        warnings.warn(f"Mavrin 2018 line radiation calculation not supported for impurity with Z={impurity_Z}", stacklevel=3)
        return np.nan

    # If trying to evaluate for a temperature outside of the given range, assume nearest neighbor
    # and throw a warning
    if any(electron_density_profile < 0.1) or any(electron_density_profile > 100):  # pragma: no cover
        warnings.warn(
            "Mavrin 2018 line radiation calculation is only valid between 0.1-100keV. Using nearest neighbor extrapolation.", stacklevel=3
        )
    electron_density_profile = np.maximum(electron_density_profile, 0.1)
    electron_density_profile = np.minimum(electron_density_profile, 100)

    # L_z coefficients for the 11 supported impurities
    if impurity_Z == 2:  # Helium
        temperature_bin_borders = np.array([0.0, 100.0])
        radc = np.array(
            [
                [-3.5551e01, 3.1469e-01, 1.0156e-01, -9.3730e-02, 2.5020e-02],
            ]
        )

    elif impurity_Z == 3:  # Lithium
        temperature_bin_borders = np.array([0.0, 100.0])
        radc = np.array(
            [
                [-3.5115e01, 1.9475e-01, 2.5082e-01, -1.6070e-01, 3.5190e-02],
            ]
        )

    elif impurity_Z == 4:  # Beryllium
        temperature_bin_borders = np.array([0.0, 100.0])
        radc = np.array(
            [
                [-3.4765e01, 3.7270e-02, 3.8363e-01, -2.1384e-01, 4.1690e-02],
            ]
        )

    elif impurity_Z == 6:  # Carbon
        temperature_bin_borders = np.array([0.0, 0.5, 100.0])
        radc = np.array(
            [
                [-3.4738e01, -5.0085e00, -1.2788e01, -1.6637e01, -7.2904e00],
                [-3.4174e01, -3.6687e-01, 6.8856e-01, -2.9191e-01, 4.4470e-02],
            ]
        )

    elif impurity_Z == 7:  # Nitrogen
        temperature_bin_borders = np.array([0.0, 0.5, 2.0, 100.0])
        radc = np.array(
            [
                [-3.4065e01, -2.3614e00, -6.0605e00, -1.1570e01, -6.9621e00],
                [-3.3899e01, -5.9668e-01, 7.6272e-01, -1.7160e-01, 5.8770e-02],
                [-3.3913e01, -5.2628e-01, 7.0047e-01, -2.2790e-01, 2.8350e-02],
            ]
        )

    elif impurity_Z == 8:  # Oxygen
        temperature_bin_borders = np.array([0.0, 0.3, 100.0])
        radc = np.array(
            [
                [-3.7257e01, -1.5635e01, -1.7141e01, -5.3765e00, 0.0000e00],
                [-3.3640e01, -7.6211e-01, 7.9655e-01, -2.0850e-01, 1.4360e-02],
            ]
        )

    elif impurity_Z == 10:  # Neon
        temperature_bin_borders = np.array([0.0, 0.7, 5, 100.0])
        radc = np.array(
            [
                [-3.3132e01, 1.7309e00, 1.5230e01, 2.8939e01, 1.5648e01],
                [-3.3290e01, -8.7750e-01, 8.6842e-01, -3.9544e-01, 1.7244e-01],
                [-3.3410e01, -4.5345e-01, 2.9731e-01, 4.3960e-02, -2.6930e-02],
            ]
        )

    elif impurity_Z == 18:  # Argon
        temperature_bin_borders = np.array([0.0, 0.6, 3, 100.0])
        radc = np.array(
            [
                [-3.2155e01, 6.5221e00, 3.0769e01, 3.9161e01, 1.5353e01],
                [-3.2530e01, 5.4490e-01, 1.5389e00, -7.6887e00, 4.9806e00],
                [-3.1853e01, -1.6674e00, 6.1339e-01, 1.7480e-01, -8.2260e-02],
            ]
        )

    elif impurity_Z == 36:  # Krypton
        temperature_bin_borders = np.array([0.0, 0.447, 2.364, 100.0])
        radc = np.array(
            [
                [-3.4512e01, -2.1484e01, -4.4723e01, -4.0133e01, -1.3564e01],
                [-3.1399e01, -5.0091e-01, 1.9148e00, -2.5865e00, -5.2704e00],
                [-2.9954e01, -6.3683e00, 6.6831e00, -2.9674e00, 4.8356e-01],
            ]
        )

    elif impurity_Z == 54:  # Xenon
        temperature_bin_borders = np.array([0.0, 0.5, 2.5, 10, 100.0])
        radc = np.array(
            [
                [-2.9303e01, 1.4351e01, 4.7081e01, 5.9580e01, 2.5615e01],
                [-3.1113e01, 5.9339e-01, 1.2808e00, -1.1628e01, 1.0748e01],
                [-2.5813e01, -2.7526e01, 4.8614e01, -3.6885e01, 1.0069e01],
                [-2.2138e01, -2.2592e01, 1.9619e01, -7.5181e00, 1.0858e00],
            ]
        )

    elif impurity_Z == 74:  # Tungsten
        temperature_bin_borders = np.array([0.0, 1.5, 4, 100.0])
        radc = np.array(
            [
                [-3.0374e01, 3.8304e-01, -9.5126e-01, -1.0311e00, -1.0103e-01],
                [-3.0238e01, -2.9208e00, 2.2824e01, -6.3303e01, 5.1849e01],
                [-3.2153e01, 5.2499e00, -6.2740e00, 2.6627e00, -3.6759e-01],
            ]
        )

    # solve for radiated power

    Tlog = np.log10(electron_density_profile)
    log10_Lz = np.zeros(electron_density_profile.size)

    for i in range(len(radc)):
        it = np.nonzero(
            (electron_density_profile >= temperature_bin_borders[i]) & (electron_density_profile < temperature_bin_borders[i + 1])
        )[0]
        if it.size > 0:
            log10_Lz[it] = polyval(Tlog[it], radc[i])  # type: ignore[no-untyped-call]

    radrate = 10.0**log10_Lz
    radrate[np.isnan(radrate)] = 0

    # 1e38 factor to account for the fact that our n_e values are electron_density_profile values
    qRad = radrate * electron_temp_profile * electron_temp_profile * impurity_concentration * 1e38  # W / (m^3 s)
    radiated_power = integrate_profile_over_volume(qRad, rho, plasma_volume)  # [W]

    return float(radiated_power) / 1e6  # MW
