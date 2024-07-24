"""Analytical functions for plasma external inductance and vertical field."""

import numpy as np
import xarray as xr

from ....unit_handling import Unitfull
from ....unit_handling import dimensionless_magnitude as dmag


def calc_fa_Sums_Na(
    inverse_aspect_ratio: Unitfull, coeffs: dict[str, np.ndarray]
) -> tuple[Unitfull, Unitfull]:
    """Calculate the sums for equation 17 on page 6 in Barr (2018).

    A power-balance model for local helicity injection startup in a spherical tokamak :cite:`Barr_2018`
    NOTE: Default values for for the coefficients 'N[a,d]' and '[a,e]' are taken from `Barr_2018` which are obtained
    from fitting them to model flux_PF and flux_Le obtained from over 330 model equilibria spanning 0<=delta<=0.5 whereas
    SPARC is projected to have delta95 = 0.54

    Args:
        inverse_aspect_ratio: [~] :term:`glossary link<inverse_aspect_ratio>`
        coeffs: coefficients from set_surface_inductance_coefficients

    Returns:
        tuple[Unitfull, Unitfull]: Calculated sums.
    """
    a = coeffs["a"]
    m = xr.DataArray(np.arange(1, 3), dims="dim_m")
    sqrt_aspect_ratio = np.sqrt(inverse_aspect_ratio)

    def func1(sqrt_eps: float) -> float:
        return float(np.sum(a[:2] * sqrt_eps**m))

    def func2(sqrt_eps: float) -> float:
        return float(np.sum(a[2:] * sqrt_eps**m))

    sum1 = xr.apply_ufunc(
        func1, dmag(sqrt_aspect_ratio), vectorize=True
    )  # type:ignore[arg-type]
    sum2 = xr.apply_ufunc(
        func2, dmag(sqrt_aspect_ratio), vectorize=True
    )  # type:ignore[arg-type]

    return sum1, sum2


def calc_fa_Sum_Ne(
    inverse_aspect_ratio: Unitfull, coeffs: dict[str, np.ndarray]
) -> Unitfull:
    """Calculate a sum for eq. 17 on page 6 in :cite:`Barr_2018`.

    A power-balance model for local helicity injection startup in a spherical tokamak :cite:`Barr_2018`

    Args:
        inverse_aspect_ratio: [~] :term:`glossary link<inverse_aspect_ratio>`
        coeffs: coefficients from set_surface_inductance_coefficients

    Returns:
         functional term [~]
    """
    e = coeffs["e"]
    m = np.arange(1, 5)

    def func(epsilon: float) -> float:
        return float(np.sum(e * np.sqrt(epsilon) ** m))

    return xr.apply_ufunc(func, dmag(inverse_aspect_ratio), vectorize=True)


def calc_fb_Sum_Nb(
    inverse_aspect_ratio: Unitfull, coeffs: dict[str, np.ndarray]
) -> Unitfull:
    """Calculate the sum for eq. 18 on page 6 in :cite:`Barr_2018`.

    A power-balance model for local helicity injection startup in a spherical tokamak :cite:`Barr_2018`


    Args:
        inverse_aspect_ratio: [~] :term:`glossary link<inverse_aspect_ratio>`
        coeffs: coefficients from set_surface_inductance_coefficients

    Returns:
         functional term [~]
    """
    b = coeffs["b"]
    n = np.arange(1, 4)

    def func(epsilon: float) -> float:
        return float(np.sum(b[n] * epsilon ** (3 + n)))

    return xr.apply_ufunc(func, dmag(inverse_aspect_ratio), vectorize=True)


def calc_fc_Sum_Nc(
    inverse_aspect_ratio: Unitfull, coeffs: dict[str, np.ndarray]
) -> Unitfull:
    """Calculate the sum for eq. 18 on page 6 in :cite:`Barr_2018`.

    A power-balance model for local helicity injection startup in a spherical tokamak :cite:`Barr_2018`

    Args:
        inverse_aspect_ratio: [~] :term:`glossary link<inverse_aspect_ratio>`
        coeffs: coefficients from set_surface_inductance_coefficients

    Returns:
         functional term [~]
    """
    c = coeffs["c"]
    m = np.arange(1, 4)

    def func(eps: float) -> float:
        return float(np.sum(c * eps ** (2 * m)))

    return xr.apply_ufunc(func, dmag(inverse_aspect_ratio), vectorize=True)


def calc_fd_Sum_Nd(
    inverse_aspect_ratio: Unitfull, coeffs: dict[str, np.ndarray]
) -> Unitfull:
    """Calculate the sum for eq. 20 on page 6 in :cite:`Barr_2018`.

    A power-balance model for local helicity injection startup in a spherical tokamak :cite:`Barr_2018`

    Args:
        inverse_aspect_ratio: [~] :term:`glossary link<inverse_aspect_ratio>`
        coeffs: coefficients from set_surface_inductance_coefficients

    Returns:
         functional term [~]
    """
    d = coeffs["d"]
    m = np.arange(2, 4)

    def func(eps: float) -> float:
        return float(np.sum(d[1:] * eps ** (m - 1)))

    return xr.apply_ufunc(func, dmag(inverse_aspect_ratio), vectorize=True)


def calc_fg_Sums_Na(
    inverse_aspect_ratio: Unitfull, coeffs: dict[str, np.ndarray]
) -> Unitfull:
    """Calculate sums for eq. 22 on page 6 in :cite:`Barr_2018`.

    A power-balance model for local helicity injection startup in a spherical tokamak :cite:`Barr_2018`

    Args:
        inverse_aspect_ratio: [~] :term:`glossary link<inverse_aspect_ratio>`
        coeffs: coefficients from set_surface_inductance_coefficients

    Returns:
         functional term [~]
    """
    a = coeffs["a"]

    sum1 = 0.5 * a[0] * (np.sqrt(inverse_aspect_ratio) ** -1) + a[1]
    sum2 = (a[0] + 0.5 * a[2]) * (1 / np.sqrt(inverse_aspect_ratio)) + (a[1] + a[3])

    return sum1, sum2


def calc_fg_Sum_Ce(
    inverse_aspect_ratio: Unitfull, coeffs: dict[str, np.ndarray]
) -> Unitfull:
    """Calculate a sum for eq. 22 on page 6 in :cite:`Barr_2018`.

    A power-balance model for local helicity injection startup in a spherical tokamak :cite:`Barr_2018`

    Args:
        inverse_aspect_ratio: [~] :term:`glossary link<inverse_aspect_ratio>`
        coeffs: coefficients from set_surface_inductance_coefficients

    Returns:
         functional term [~]
    """
    e = coeffs["e"]
    m = np.arange(1, 5)

    def func(epsilon: float) -> float:
        return float(np.sum(m / 2 * e * np.sqrt(epsilon) ** (m - 1)))

    return xr.apply_ufunc(func, dmag(inverse_aspect_ratio), vectorize=True)


def calc_fh_Sum_Cb(
    inverse_aspect_ratio: Unitfull, coeffs: dict[str, np.ndarray]
) -> Unitfull:
    """Calculate a sum for eq. 23 on page 6 in :cite:`Barr_2018`.

    A power-balance model for local helicity injection startup in a spherical tokamak :cite:`Barr_2018`

    Args:
        inverse_aspect_ratio: [~] :term:`glossary link<inverse_aspect_ratio>`
        coeffs: coefficients from set_surface_inductance_coefficients

    Returns:
         functional term [~]
    """
    b = coeffs["b"]
    n = np.arange(1, 4)

    def func(epsilon: float) -> float:
        return float(np.sum((n + 3.5) * b[n] * epsilon ** (n + 3)))

    return xr.apply_ufunc(func, dmag(inverse_aspect_ratio), vectorize=True)


def calc_fa(
    inverse_aspect_ratio: Unitfull,
    beta_poloidal: Unitfull,
    internal_inductivity: Unitfull,
    coeffs: dict[str, np.ndarray],
) -> Unitfull:
    """Calculate eq. 17 on page 6 in :cite:`Barr_2018`.

    A power-balance model for local helicity injection startup in a spherical tokamak :cite:`Barr_2018`

    Args:
        inverse_aspect_ratio: [~] :term:`glossary link<inverse_aspect_ratio>`
        beta_poloidal: [~] :term:`glossary link<beta_poloidal>`
        internal_inductivity: [~] :term:`glossary link<internal_inductivity>`
        coeffs: coefficients from set_surface_inductance_coefficients

    Returns:
         functional term [~]
    """
    fa_Sums_ = calc_fa_Sums_Na(inverse_aspect_ratio, coeffs=coeffs)

    return (
        ((1 + fa_Sums_[0]) * np.log(8 / inverse_aspect_ratio))
        - (2 + fa_Sums_[1])
        + (beta_poloidal + internal_inductivity / 2)
        * calc_fa_Sum_Ne(inverse_aspect_ratio, coeffs=coeffs)
    )


def calc_fb(inverse_aspect_ratio: Unitfull, coeffs: dict[str, np.ndarray]) -> Unitfull:
    """Calculate eq. 18 on page 6 in :cite:`Barr_2018`.

    A power-balance model for local helicity injection startup in a spherical tokamak :cite:`Barr_2018`

    Args:
        inverse_aspect_ratio: [~] :term:`glossary link<inverse_aspect_ratio>`
        coeffs: coefficients from set_surface_inductance_coefficients

    Returns:
         functional term [~]
    """
    b = coeffs["b"]

    return (
        b[0]
        * np.sqrt(inverse_aspect_ratio)
        * (1 + calc_fb_Sum_Nb(inverse_aspect_ratio, coeffs=coeffs))
    )


def calc_fc(inverse_aspect_ratio: Unitfull, coeffs: dict[str, np.ndarray]) -> Unitfull:
    """Calculate eq. 19 on page 6 in :cite:`Barr_2018`.

    A power-balance model for local helicity injection startup in a spherical tokamak :cite:`Barr_2018`

    Args:
        inverse_aspect_ratio: [~] :term:`glossary link<inverse_aspect_ratio>`
        coeffs: coefficients from set_surface_inductance_coefficients

    Returns:
         functional term [~]
    """
    fc = 1 + calc_fc_Sum_Nc(inverse_aspect_ratio, coeffs=coeffs)

    return fc


def calc_fd(inverse_aspect_ratio: Unitfull, coeffs: dict[str, np.ndarray]) -> Unitfull:
    """Calculate eq. 20 on page 6 in :cite:`Barr_2018`.

    A power-balance model for local helicity injection startup in a spherical tokamak :cite:`Barr_2018`

    Args:
        inverse_aspect_ratio: [~] :term:`glossary link<inverse_aspect_ratio>`
        areal_elongation: [~] :term:`glossary link<areal_elongation>`
        coeffs: coefficients from set_surface_inductance_coefficients

    Returns:
         functional term [~]
    """
    d = coeffs["d"]
    fd = (
        d[0]
        * inverse_aspect_ratio
        * (1 + calc_fd_Sum_Nd(inverse_aspect_ratio, coeffs=coeffs))
    )

    return fd


def calc_fg(
    inverse_aspect_ratio: Unitfull,
    beta_poloidal: Unitfull,
    internal_inductivity: Unitfull,
    coeffs: dict[str, np.ndarray],
) -> Unitfull:
    """Calculate eq. 22 on page 6 in :cite:`Barr_2018`.

    A power-balance model for local helicity injection startup in a spherical tokamak :cite:`Barr_2018`

    Args:
        inverse_aspect_ratio: [~] :term:`glossary link<inverse_aspect_ratio>`
        beta_poloidal: [~] :term:`glossary link<beta_poloidal>`
        internal_inductivity: [~] :term:`glossary link<internal_inductivity>`
        coeffs: coefficients from set_surface_inductance_coefficients

    Returns:
         functional term [~]
    """
    fg_Sums_ = calc_fg_Sums_Na(inverse_aspect_ratio, coeffs=coeffs)

    return (
        -(1 / inverse_aspect_ratio)
        + np.log(8 / inverse_aspect_ratio) * fg_Sums_[0]
        - fg_Sums_[1]
        + (beta_poloidal + (internal_inductivity / 2))
        * calc_fg_Sum_Ce(inverse_aspect_ratio, coeffs=coeffs)
    )


def calc_fh(
    inverse_aspect_ratio: Unitfull,
    areal_elongation: Unitfull,
    coeffs: dict[str, np.ndarray],
) -> Unitfull:
    """Calculate eq. 23 on page 6 in :cite:`Barr_2018`.

    A power-balance model for local helicity injection startup in a spherical tokamak :cite:`Barr_2018`

    Args:
        inverse_aspect_ratio: [~] :term:`glossary link<inverse_aspect_ratio>`
        areal_elongation: [~] :term:`glossary link<areal_elongation>`
        coeffs: coefficients from set_surface_inductance_coefficients

    Returns:
         functional term [~]
    """
    b = coeffs["b"]

    return -1 + ((areal_elongation * b[0]) / np.sqrt(inverse_aspect_ratio)) * (
        1 / 2 + calc_fh_Sum_Cb(inverse_aspect_ratio, coeffs=coeffs)
    )
