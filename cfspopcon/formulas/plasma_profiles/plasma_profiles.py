"""Estimate 1D plasma profiles of density and temperature."""

from bisect import bisect_left
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.optimize import brentq
from ...algorithm_class import Algorithm
from ...named_options import ProfileForm
from ...unit_handling import Unitfull, ureg, wraps_ufunc
from .density_peaking import calc_density_peaking, calc_effective_collisionality
from .numerical_profile_fits import evaluate_density_and_temperature_profile_fits


@Algorithm.register_algorithm(
    return_keys=[
        "effective_collisionality",
        "ion_density_peaking",
        "ion_density_pedestal_peaking",
        "electron_density_peaking",
        "electron_density_pedestal_peaking",
        "electron_temp_pedestal_peaking",
        "ion_temp_pedestal_peaking",
        "peak_electron_density",
        "peak_fuel_ion_density",
        "peak_electron_temp",
        "peak_ion_temp",
        "rho",
        "electron_density_profile",
        "fuel_ion_density_profile",
        "electron_temp_profile",
        "ion_temp_profile",
    ]
)
def calc_peaked_profiles(
    average_electron_density: Unitfull,
    average_electron_temp: Unitfull,
    average_ion_temp: Unitfull,
    ion_density_peaking_offset: Unitfull,
    electron_density_peaking_offset: Unitfull,
    temperature_peaking: Unitfull,
    major_radius: Unitfull,
    z_effective: Unitfull,
    dilution: Unitfull,
    beta_toroidal: Unitfull,
    normalized_inverse_temp_scale_length: Unitfull,
    density_profile_form: ProfileForm,
    temp_profile_form: ProfileForm,
    pedestal_width: Unitfull = 0.05 * ureg.dimensionless,
    t_sep: Unitfull = 0.2 * ureg.keV,
    n_sep_ratio: Unitfull = 0.5 * ureg.dimensionless,
) -> tuple[Unitfull, ...]:
    """Calculate density peaking and the corresponding density and temperature profiles.

    Args:
        average_electron_density: :term:`glossary link<average_electron_density>`
        average_electron_temp: :term:`glossary link<average_electron_temp>`
        average_ion_temp: :term:`glossary link<average_ion_temp>`
        ion_density_peaking_offset: :term:`glossary link<ion_density_peaking_offset>`
        electron_density_peaking_offset: :term:`glossary link<electron_density_peaking_offset>`
        temperature_peaking: :term:`glossary link<temperature_peaking>`
        major_radius: :term:`glossary link<major_radius>`
        z_effective: :term:`glossary link<z_effective>`
        dilution: :term:`glossary link<dilution>`
        beta_toroidal: :term:`glossary link<beta_toroidal>`
        normalized_inverse_temp_scale_length: :term:`glossary link<normalized_inverse_temp_scale_length>`
        density_profile_form: :term:`glossary link<density_profile_form>`
        temp_profile_form: :term:`glossary link<temp_profile_form>`
        pedestal_width: Pedestal width in normalized rho.
        t_sep: Separatrix temperature used to anchor the JCH edge temperature profile.
        n_sep_ratio: Ratio of separatrix density to pedestal density for JCH profiles.

    Returns:
        :term:`effective_collisionality`, :term:`ion_density_peaking`, :term:`electron_density_peaking`,
        :term:`ion_density_pedestal_peaking`, :term:`electron_density_pedestal_peaking`,
        :term:`electron_temp_pedestal_peaking`, :term:`ion_temp_pedestal_peaking`,
        :term:`peak_electron_density`, :term:`peak_fuel_ion_density`, :term:`peak_electron_temp`,
        :term:`peak_ion_temp`, :term:`rho`, :term:`electron_density_profile`, :term:`fuel_ion_density_profile`,
        :term:`electron_temp_profile`, :term:`ion_temp_profile`
    """
    effective_collisionality = calc_effective_collisionality(average_electron_density, average_electron_temp, major_radius, z_effective)
    ion_density_peaking = calc_density_peaking(effective_collisionality, beta_toroidal, nu_noffset=ion_density_peaking_offset)
    electron_density_peaking = calc_density_peaking(effective_collisionality, beta_toroidal, nu_noffset=electron_density_peaking_offset)
    peak_electron_density = average_electron_density * electron_density_peaking
    peak_fuel_ion_density = average_electron_density * dilution * ion_density_peaking
    peak_electron_temp = average_electron_temp * temperature_peaking
    peak_ion_temp = average_ion_temp * temperature_peaking

    # Calculate the total fusion power by estimating density and temperature profiles and
    # using this to calculate fusion power profiles.
    (rho, electron_density_profile, fuel_ion_density_profile, electron_temp_profile, ion_temp_profile) = calc_1D_plasma_profiles(
        density_profile_form=density_profile_form,
        temp_profile_form=temp_profile_form,
        average_electron_density=average_electron_density,
        average_electron_temp=average_electron_temp,
        average_ion_temp=average_ion_temp,
        electron_density_peaking=electron_density_peaking,
        ion_density_peaking=ion_density_peaking,
        temperature_peaking=temperature_peaking,
        dilution=dilution,
        normalized_inverse_temp_scale_length=normalized_inverse_temp_scale_length,
        pedestal_width=pedestal_width,
        t_sep=t_sep,
        n_sep_ratio=n_sep_ratio,
    )

    if density_profile_form == ProfileForm.jch or temp_profile_form == ProfileForm.jch:
        (
            electron_density_pedestal_peaking,
            ion_density_pedestal_peaking,
            electron_temp_pedestal_peaking,
            ion_temp_pedestal_peaking,
        ) = calc_jch_pedestal_peaking(
            density_profile_form=density_profile_form,
            temp_profile_form=temp_profile_form,
            average_electron_temp=average_electron_temp,
            average_ion_temp=average_ion_temp,
            electron_density_peaking=electron_density_peaking,
            ion_density_peaking=ion_density_peaking,
            temperature_peaking=temperature_peaking,
            pedestal_width=pedestal_width,
            t_sep=t_sep,
            n_sep_ratio=n_sep_ratio,
        )
    else:
        electron_density_pedestal_peaking = electron_density_peaking * np.nan
        ion_density_pedestal_peaking = ion_density_peaking * np.nan
        electron_temp_pedestal_peaking = temperature_peaking * np.nan
        ion_temp_pedestal_peaking = temperature_peaking * np.nan

    return (
        effective_collisionality,
        ion_density_peaking,
        ion_density_pedestal_peaking,
        electron_density_peaking,
        electron_density_pedestal_peaking,
        electron_temp_pedestal_peaking,
        ion_temp_pedestal_peaking,
        peak_electron_density,
        peak_fuel_ion_density,
        peak_electron_temp,
        peak_ion_temp,
        rho,
        electron_density_profile,
        fuel_ion_density_profile,
        electron_temp_profile,
        ion_temp_profile,
    )


@Algorithm.register_algorithm(
    return_keys=["rho", "electron_density_profile", "fuel_ion_density_profile", "electron_temp_profile", "ion_temp_profile"]
)
@wraps_ufunc(
    return_units=dict(
        rho=ureg.dimensionless,
        electron_density_profile=ureg.n19,
        fuel_ion_density_profile=ureg.n19,
        electron_temp_profile=ureg.keV,
        ion_temp_profile=ureg.keV,
    ),
    input_units=dict(
        density_profile_form=None,
        temp_profile_form=None,
        average_electron_density=ureg.n19,
        average_electron_temp=ureg.keV,
        average_ion_temp=ureg.keV,
        electron_density_peaking=ureg.dimensionless,
        ion_density_peaking=ureg.dimensionless,
        temperature_peaking=ureg.dimensionless,
        dilution=ureg.dimensionless,
        normalized_inverse_temp_scale_length=ureg.dimensionless,
        n_points_for_confined_region_profiles=None,
        pedestal_width=ureg.dimensionless,
        t_sep=ureg.keV,
        n_sep_ratio=ureg.dimensionless,
    ),
    output_core_dims=[("dim_rho",), ("dim_rho",), ("dim_rho",), ("dim_rho",), ("dim_rho",)],
)
def calc_1D_plasma_profiles(
    density_profile_form: ProfileForm,
    temp_profile_form: ProfileForm,
    average_electron_density: float,
    average_electron_temp: float,
    average_ion_temp: float,
    electron_density_peaking: float,
    ion_density_peaking: float,
    temperature_peaking: float,
    dilution: float,
    normalized_inverse_temp_scale_length: float,
    n_points_for_confined_region_profiles: int = 50,
    pedestal_width: float = 0.05 * ureg.dimensionless,
    t_sep: float = 0.2 * ureg.keV,
    n_sep_ratio: float = 0.5 * ureg.dimensionless,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate density and temperature profiles.

    The peaking inputs are interpreted as peak-to-volume-average ratios for all
    profile forms. When JCH profiles are requested, those volume peaking values
    are converted internally to the peak-to-pedestal ratios required by the
    JCH parameterization.

    Args:
        density_profile_form: :term:`glossary link<density_profile_form>`
        temp_profile_form: :term:`glossary link<temp_profile_form>`
        average_electron_density: [1e19 m^-3] :term:`glossary link<average_electron_density>`
        average_electron_temp: [keV] :term:`glossary link<average_electron_temp>`
        average_ion_temp: [keV] :term:`glossary link<average_ion_temp>`
        electron_density_peaking: [~] :term:`glossary link<electron_density_peaking>`
        ion_density_peaking: [~] :term:`glossary link<ion_density_peaking>`
        temperature_peaking: [~] :term:`glossary link<temperature_peaking>`
        dilution: dilution of main ions [~]
        normalized_inverse_temp_scale_length: [~] :term:`glossary link<normalized_inverse_temp_scale_length>`
        n_points_for_confined_region_profiles: Number of points to return in the profile grid.
        pedestal_width: Pedestal width in normalized rho for JCH profiles.
        t_sep: Separatrix temperature used to anchor the JCH edge temperature profile.
        n_sep_ratio: Ratio of separatrix density to pedestal density for JCH profiles.

    Returns:
        :term:`rho` [~], :term:`electron_density_profile` [1e19 m^-3],
        :term:`fuel_ion_density_profile` [1e19 m^-3], :term:`electron_temp_profile` [keV],
        :term:`ion_temp_profile` [keV]
    """
    needs_jch_profiles = density_profile_form == ProfileForm.jch or temp_profile_form == ProfileForm.jch
    rho_grid = _build_profile_grid(
        n_points_for_confined_region_profiles,
        (1.0 - float(pedestal_width)) if needs_jch_profiles else None,
    )

    kwargs: dict[str, Any] = dict(
        average_electron_density=average_electron_density,
        average_electron_temp=average_electron_temp,
        average_ion_temp=average_ion_temp,
        electron_density_peaking=electron_density_peaking,
        ion_density_peaking=ion_density_peaking,
        temperature_peaking=temperature_peaking,
        dilution=dilution,
        npoints=n_points_for_confined_region_profiles,
        rho=rho_grid,
    )

    electron_density_profiles, fuel_ion_density_profiles, electron_temp_profiles, ion_temp_profiles = dict(), dict(), dict(), dict()
    (
        rho_1,
        electron_density_profiles[ProfileForm.analytic],
        fuel_ion_density_profiles[ProfileForm.analytic],
        electron_temp_profiles[ProfileForm.analytic],
        ion_temp_profiles[ProfileForm.analytic],
    ) = calc_analytic_profiles(**kwargs)

    (
        rho_2,
        electron_density_profiles[ProfileForm.prf],
        fuel_ion_density_profiles[ProfileForm.prf],
        electron_temp_profiles[ProfileForm.prf],
        ion_temp_profiles[ProfileForm.prf],
    ) = calc_prf_profiles(**kwargs, normalized_inverse_temp_scale_length=normalized_inverse_temp_scale_length)

    if needs_jch_profiles:
        (
            rho_3,
            electron_density_profile_jch,
            fuel_ion_density_profile_jch,
            electron_temp_profile_jch,
            ion_temp_profile_jch,
        ) = calc_jch_profiles(
            average_electron_density=average_electron_density,
            average_electron_temp=average_electron_temp,
            average_ion_temp=average_ion_temp,
            electron_density_volume_peaking=electron_density_peaking if density_profile_form == ProfileForm.jch else None,
            ion_density_volume_peaking=ion_density_peaking if density_profile_form == ProfileForm.jch else None,
            electron_temp_volume_peaking=temperature_peaking if temp_profile_form == ProfileForm.jch else None,
            ion_temp_volume_peaking=temperature_peaking if temp_profile_form == ProfileForm.jch else None,
            dilution=dilution,
            n_points=n_points_for_confined_region_profiles,
            pedestal_width=pedestal_width,
            t_sep=t_sep,
            n_sep_ratio=n_sep_ratio,
            rho=rho_grid,
        )
        if density_profile_form == ProfileForm.jch:
            assert electron_density_profile_jch is not None
            assert fuel_ion_density_profile_jch is not None
            electron_density_profiles[ProfileForm.jch] = electron_density_profile_jch
            fuel_ion_density_profiles[ProfileForm.jch] = fuel_ion_density_profile_jch
        if temp_profile_form == ProfileForm.jch:
            assert electron_temp_profile_jch is not None
            assert ion_temp_profile_jch is not None
            electron_temp_profiles[ProfileForm.jch] = electron_temp_profile_jch
            ion_temp_profiles[ProfileForm.jch] = ion_temp_profile_jch

    assert np.allclose(rho_1, rho_2)
    if needs_jch_profiles:
        assert np.allclose(rho_1, rho_3)

    return (
        rho_1,
        electron_density_profiles[density_profile_form],
        fuel_ion_density_profiles[density_profile_form],
        electron_temp_profiles[temp_profile_form],
        ion_temp_profiles[temp_profile_form],
    )


@wraps_ufunc(
    return_units=dict(
        electron_density_pedestal_peaking=ureg.dimensionless,
        ion_density_pedestal_peaking=ureg.dimensionless,
        electron_temp_pedestal_peaking=ureg.dimensionless,
        ion_temp_pedestal_peaking=ureg.dimensionless,
    ),
    input_units=dict(
        density_profile_form=None,
        temp_profile_form=None,
        average_electron_temp=ureg.keV,
        average_ion_temp=ureg.keV,
        electron_density_peaking=ureg.dimensionless,
        ion_density_peaking=ureg.dimensionless,
        temperature_peaking=ureg.dimensionless,
        n_points_for_confined_region_profiles=None,
        pedestal_width=ureg.dimensionless,
        t_sep=ureg.keV,
        n_sep_ratio=ureg.dimensionless,
    ),
    output_core_dims=[(), (), (), ()],
)
def calc_jch_pedestal_peaking(
    density_profile_form: ProfileForm,
    temp_profile_form: ProfileForm,
    average_electron_temp: float,
    average_ion_temp: float,
    electron_density_peaking: float,
    ion_density_peaking: float,
    temperature_peaking: float,
    n_points_for_confined_region_profiles: int = 50,
    pedestal_width: float = 0.05 * ureg.dimensionless,
    t_sep: float = 0.2 * ureg.keV,
    n_sep_ratio: float = 0.5 * ureg.dimensionless,
) -> tuple[float, float, float, float]:
    """Convert volume peaking values into the JCH peak-to-pedestal ratios."""
    if density_profile_form != ProfileForm.jch and temp_profile_form != ProfileForm.jch:
        return np.nan, np.nan, np.nan, np.nan

    pedestal_width = float(pedestal_width)
    rho_ped = 1.0 - pedestal_width
    rho = _build_profile_grid(int(n_points_for_confined_region_profiles), rho_ped)
    rho_core = rho[rho <= rho_ped]
    _, _, edge_integral_1, edge_integral_2 = _calc_jch_edge_integrals(rho, rho_ped, pedestal_width)

    electron_density_pedestal_peaking = np.nan
    ion_density_pedestal_peaking = np.nan
    if density_profile_form == ProfileForm.jch:
        electron_density_pedestal_peaking = _solve_jch_density_pedestal_peaking(
            target_volume_peaking=float(electron_density_peaking),
            rho_core=rho_core,
            rho_ped=rho_ped,
            edge_integral_1=edge_integral_1,
            edge_integral_2=edge_integral_2,
            separatrix_to_pedestal_ratio=float(n_sep_ratio),
        )
        ion_density_pedestal_peaking = _solve_jch_density_pedestal_peaking(
            target_volume_peaking=float(ion_density_peaking),
            rho_core=rho_core,
            rho_ped=rho_ped,
            edge_integral_1=edge_integral_1,
            edge_integral_2=edge_integral_2,
            separatrix_to_pedestal_ratio=float(n_sep_ratio),
        )

    electron_temp_pedestal_peaking = np.nan
    ion_temp_pedestal_peaking = np.nan
    if temp_profile_form == ProfileForm.jch:
        electron_temp_pedestal_peaking = _solve_jch_temperature_pedestal_peaking(
            target_volume_peaking=float(temperature_peaking),
            volume_average=float(average_electron_temp),
            rho_core=rho_core,
            rho_ped=rho_ped,
            edge_integral_1=edge_integral_1,
            edge_integral_2=edge_integral_2,
            separatrix_temperature=float(t_sep),
        )
        ion_temp_pedestal_peaking = _solve_jch_temperature_pedestal_peaking(
            target_volume_peaking=float(temperature_peaking),
            volume_average=float(average_ion_temp),
            rho_core=rho_core,
            rho_ped=rho_ped,
            edge_integral_1=edge_integral_1,
            edge_integral_2=edge_integral_2,
            separatrix_temperature=float(t_sep),
        )

    return (
        electron_density_pedestal_peaking,
        ion_density_pedestal_peaking,
        electron_temp_pedestal_peaking,
        ion_temp_pedestal_peaking,
    )


def calc_analytic_profiles(
    average_electron_density: float,
    average_electron_temp: float,
    average_ion_temp: float,
    electron_density_peaking: float,
    ion_density_peaking: float,
    temperature_peaking: float,
    dilution: float,
    npoints: int = 50,
    rho: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate density and temperature profiles using a simple analytic fit.

    Args:
        average_electron_density: [1e19 m^-3] :term:`glossary link<average_electron_density>`
        average_electron_temp: [keV] :term:`glossary link<average_electron_temp>`
        average_ion_temp: [keV] :term:`glossary link<average_ion_temp>`
        electron_density_peaking: [~] :term:`glossary link<electron_density_peaking>`
        ion_density_peaking: [~] :term:`glossary link<ion_density_peaking>`
        temperature_peaking: [~] :term:`glossary link<temperature_peaking>`
        dilution: dilution of main ions [~]
        npoints: number of points to return in profile

    Returns:
        :term:`rho` [~], :term:`electron_density_profile` [1e19 m^-3], fuel_ion_density_profile [1e19 m^-3], :term:`electron_temp_profile` [keV], :term:`ion_temp_profile` [keV]
    """
    if rho is None:
        rho = _build_profile_grid(npoints)
    else:
        rho = np.asarray(rho, dtype=float)

    electron_density_profile = average_electron_density * electron_density_peaking * ((1.0 - rho**2.0) ** (electron_density_peaking - 1.0))
    fuel_ion_density_profile = (
        average_electron_density * dilution * (ion_density_peaking) * ((1.0 - rho**2.0) ** (ion_density_peaking - 1.0))
    )
    electron_temp_profile = average_electron_temp * temperature_peaking * ((1.0 - rho**2.0) ** (temperature_peaking - 1.0))
    ion_temp_profile = average_ion_temp * temperature_peaking * ((1.0 - rho**2.0) ** (temperature_peaking - 1.0))

    return rho, electron_density_profile, fuel_ion_density_profile, electron_temp_profile, ion_temp_profile


def calc_prf_profiles(
    average_electron_density: float,
    average_electron_temp: float,
    average_ion_temp: float,
    electron_density_peaking: float,
    ion_density_peaking: float,
    temperature_peaking: float,
    dilution: float,
    normalized_inverse_temp_scale_length: float,
    npoints: int = 50,
    rho: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate density and temperature profiles using profiles from Pablo Rodriguez-Fernandez.

    Args:
        average_electron_density: [1e19 m^-3] :term:`glossary link<average_electron_density>`
        average_electron_temp: [keV] :term:`glossary link<average_electron_temp>`
        average_ion_temp: [keV] :term:`glossary link<average_ion_temp>`
        electron_density_peaking: [~] :term:`glossary link<electron_density_peaking>`
        ion_density_peaking: [~] :term:`glossary link<ion_density_peaking>`
        temperature_peaking: [~] :term:`glossary link<temperature_peaking>`
        dilution: dilution of main ions [~]
        normalized_inverse_temp_scale_length: [~] :term:`glossary link<normalized_inverse_temp_scale_length>`
        npoints: number of points to return in profile

    Returns:
        :term:`rho` [~], :term:`electron_density_profile` [1e19 m^-3], fuel_ion_density_profile [1e19 m^-3], :term:`electron_temp_profile` [keV], :term:`ion_temp_profile` [keV]
    """
    if rho is None:
        rho = _build_profile_grid(npoints)
    else:
        rho = np.asarray(rho, dtype=float)

    rho, electron_temp_profile, electron_density_profile = evaluate_density_and_temperature_profile_fits(
        average_electron_temp,
        average_electron_density,
        temperature_peaking,
        electron_density_peaking,
        aLT=normalized_inverse_temp_scale_length,
        rho=rho,
        dataset="PRF",
    )
    rho, ion_temp_profile, fuel_ion_density_profile = evaluate_density_and_temperature_profile_fits(
        average_ion_temp,
        average_electron_density * dilution,
        temperature_peaking,
        ion_density_peaking,
        aLT=normalized_inverse_temp_scale_length,
        rho=rho,
        dataset="PRF",
    )

    return rho, electron_density_profile, fuel_ion_density_profile, electron_temp_profile, ion_temp_profile


def _find_nearest_grid_index(values: np.ndarray, target: float) -> int:
    """Find the nearest point in a sorted grid using a bisection search."""
    insertion_index = bisect_left(values.tolist(), target)

    if insertion_index == 0:
        return 0
    if insertion_index == len(values):
        return len(values) - 1
    if (values[insertion_index] - target) < (target - values[insertion_index - 1]):
        return insertion_index

    return insertion_index - 1


def _find_nearest_interior_grid_index(values: np.ndarray, target: float) -> int:
    """Find the nearest interior point in a sorted grid."""
    if len(values) < 3:
        raise ValueError("JCH profiles require at least three radial grid points to preserve both the axis and separatrix.")

    return 1 + _find_nearest_grid_index(values[1:-1], target)


def _build_profile_grid(npoints: int, rho_ped: float | None = None) -> np.ndarray:
    """Build the radial grid and optionally reserve four points across the pedestal."""
    rho = np.linspace(0.0, 1.0, num=npoints)

    if rho_ped is None:
        return rho

    pedestal_points = 4
    if npoints < pedestal_points + 1:
        raise ValueError("JCH profile grids require at least five radial points to preserve the axis and four pedestal samples.")

    core_points = npoints - pedestal_points + 1
    rho_core = np.linspace(0.0, rho_ped, num=core_points)
    rho_pedestal = np.linspace(rho_ped, 1.0, num=pedestal_points)
    return np.concatenate((rho_core, rho_pedestal[1:]))


def _calc_jch_core_integral(gradient: float, rho_core: np.ndarray, rho_ped: float) -> float:
    """Integrate a pedestal-normalized exponential core profile over the confined volume."""
    profile = np.exp(gradient * (rho_ped - rho_core))
    return float(np.trapezoid(profile * 2.0 * rho_core, x=rho_core))


def _calc_jch_edge_integrals(rho: np.ndarray, rho_ped: float, pedestal_width: float) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Compute the linear pedestal basis functions and their volume integrals."""
    rho_edge = rho[rho >= rho_ped]
    edge_basis_1 = (1.0 - rho_edge) / pedestal_width
    edge_basis_2 = (rho_edge - rho_ped) / pedestal_width

    edge_integral_1 = float(np.trapezoid(edge_basis_1 * 2.0 * rho_edge, x=rho_edge))
    edge_integral_2 = float(np.trapezoid(edge_basis_2 * 2.0 * rho_edge, x=rho_edge))

    return edge_basis_1, edge_basis_2, edge_integral_1, edge_integral_2


def _calc_jch_density_volume_peaking(
    peak_to_pedestal: float,
    rho_core: np.ndarray,
    rho_ped: float,
    edge_integral_1: float,
    edge_integral_2: float,
    separatrix_to_pedestal_ratio: float,
) -> float:
    """Return the peak-to-volume-average ratio for a JCH density profile."""
    gradient = float(np.log(peak_to_pedestal) / rho_ped)
    core_integral = _calc_jch_core_integral(gradient, rho_core, rho_ped)
    return peak_to_pedestal / (core_integral + edge_integral_1 + separatrix_to_pedestal_ratio * edge_integral_2)


def _calc_jch_temperature_pedestal_temperature(
    volume_average: float,
    peak_to_pedestal: float,
    rho_core: np.ndarray,
    rho_ped: float,
    edge_integral_1: float,
    edge_integral_2: float,
    separatrix_temperature: float,
) -> float:
    """Return the pedestal temperature implied by a JCH peak-to-pedestal ratio."""
    gradient = float(np.log(peak_to_pedestal) / rho_ped)
    core_integral = _calc_jch_core_integral(gradient, rho_core, rho_ped)
    return (volume_average - separatrix_temperature * edge_integral_2) / (core_integral + edge_integral_1)


def _calc_jch_temperature_volume_peaking(
    volume_average: float,
    peak_to_pedestal: float,
    rho_core: np.ndarray,
    rho_ped: float,
    edge_integral_1: float,
    edge_integral_2: float,
    separatrix_temperature: float,
) -> float:
    """Return the peak-to-volume-average ratio for a JCH temperature profile."""
    pedestal_temperature = _calc_jch_temperature_pedestal_temperature(
        volume_average=volume_average,
        peak_to_pedestal=peak_to_pedestal,
        rho_core=rho_core,
        rho_ped=rho_ped,
        edge_integral_1=edge_integral_1,
        edge_integral_2=edge_integral_2,
        separatrix_temperature=separatrix_temperature,
    )
    return peak_to_pedestal * pedestal_temperature / volume_average


def _solve_jch_peak_to_pedestal(
    target_volume_peaking: float,
    volume_peaking_function: Callable[[float], float],
    quantity_name: str,
    maximum_peak_to_pedestal: float | None = None,
) -> float:
    """Solve for the JCH peak-to-pedestal ratio that matches a target volume peaking."""
    minimum_volume_peaking = volume_peaking_function(1.0)
    if target_volume_peaking < minimum_volume_peaking and not np.isclose(target_volume_peaking, minimum_volume_peaking):
        raise ValueError(
            f"Requested JCH {quantity_name} peaking {target_volume_peaking} is below the minimum compatible "
            f"volume peaking {minimum_volume_peaking}."
        )
    if np.isclose(target_volume_peaking, minimum_volume_peaking):
        return 1.0

    if maximum_peak_to_pedestal is None:
        lower_bound = 1.0
        upper_bound = 2.0
        while volume_peaking_function(upper_bound) < target_volume_peaking:
            lower_bound, upper_bound = upper_bound, upper_bound * 2.0
            if upper_bound > 1.0e12:
                raise ValueError(f"Could not bracket the requested JCH {quantity_name} peaking {target_volume_peaking}.")
    else:
        lower_bound = 1.0
        upper_bound = maximum_peak_to_pedestal
        maximum_volume_peaking = volume_peaking_function(upper_bound)
        if target_volume_peaking > maximum_volume_peaking and not np.isclose(target_volume_peaking, maximum_volume_peaking):
            raise ValueError(
                f"Requested JCH {quantity_name} peaking {target_volume_peaking} exceeds the maximum compatible "
                f"volume peaking {maximum_volume_peaking}."
            )
        if np.isclose(target_volume_peaking, maximum_volume_peaking):
            return upper_bound

    return float(
        brentq(lambda peak_to_pedestal: volume_peaking_function(peak_to_pedestal) - target_volume_peaking, lower_bound, upper_bound)
    )


def _solve_jch_density_pedestal_peaking(
    target_volume_peaking: float,
    rho_core: np.ndarray,
    rho_ped: float,
    edge_integral_1: float,
    edge_integral_2: float,
    separatrix_to_pedestal_ratio: float,
) -> float:
    """Convert a density peak-to-average ratio into the JCH peak-to-pedestal ratio."""
    return _solve_jch_peak_to_pedestal(
        target_volume_peaking=target_volume_peaking,
        volume_peaking_function=lambda peak_to_pedestal: _calc_jch_density_volume_peaking(
            peak_to_pedestal=peak_to_pedestal,
            rho_core=rho_core,
            rho_ped=rho_ped,
            edge_integral_1=edge_integral_1,
            edge_integral_2=edge_integral_2,
            separatrix_to_pedestal_ratio=separatrix_to_pedestal_ratio,
        ),
        quantity_name="density",
    )


def _solve_jch_temperature_pedestal_peaking(
    target_volume_peaking: float,
    volume_average: float,
    rho_core: np.ndarray,
    rho_ped: float,
    edge_integral_1: float,
    edge_integral_2: float,
    separatrix_temperature: float,
) -> float:
    """Convert a temperature peak-to-average ratio into the JCH peak-to-pedestal ratio."""
    volume_peaking_function = lambda peak_to_pedestal: _calc_jch_temperature_volume_peaking(
        volume_average=volume_average,
        peak_to_pedestal=peak_to_pedestal,
        rho_core=rho_core,
        rho_ped=rho_ped,
        edge_integral_1=edge_integral_1,
        edge_integral_2=edge_integral_2,
        separatrix_temperature=separatrix_temperature,
    )

    maximum_peak_to_pedestal: float | None = None
    if separatrix_temperature > 0.0:
        if (
            _calc_jch_temperature_pedestal_temperature(
                volume_average=volume_average,
                peak_to_pedestal=1.0,
                rho_core=rho_core,
                rho_ped=rho_ped,
                edge_integral_1=edge_integral_1,
                edge_integral_2=edge_integral_2,
                separatrix_temperature=separatrix_temperature,
            )
            < separatrix_temperature
        ):
            raise ValueError("Requested JCH temperature profile gives an unphysical pedestal below the separatrix temperature.")

        lower_bound = 1.0
        upper_bound = 2.0
        while (
            _calc_jch_temperature_pedestal_temperature(
                volume_average=volume_average,
                peak_to_pedestal=upper_bound,
                rho_core=rho_core,
                rho_ped=rho_ped,
                edge_integral_1=edge_integral_1,
                edge_integral_2=edge_integral_2,
                separatrix_temperature=separatrix_temperature,
            )
            >= separatrix_temperature
        ):
            lower_bound, upper_bound = upper_bound, upper_bound * 2.0
            if upper_bound > 1.0e12:
                raise ValueError("Could not bracket the maximum valid JCH temperature pedestal peaking.")

        maximum_peak_to_pedestal = float(
            brentq(
                lambda peak_to_pedestal: _calc_jch_temperature_pedestal_temperature(
                    volume_average=volume_average,
                    peak_to_pedestal=peak_to_pedestal,
                    rho_core=rho_core,
                    rho_ped=rho_ped,
                    edge_integral_1=edge_integral_1,
                    edge_integral_2=edge_integral_2,
                    separatrix_temperature=separatrix_temperature,
                )
                - separatrix_temperature,
                lower_bound,
                upper_bound,
            )
        )

    return _solve_jch_peak_to_pedestal(
        target_volume_peaking=target_volume_peaking,
        volume_peaking_function=volume_peaking_function,
        quantity_name="temperature",
        maximum_peak_to_pedestal=maximum_peak_to_pedestal,
    )


def _build_jch_density_profile(
    volume_average: float,
    peak_to_pedestal: float,
    rho: np.ndarray,
    rho_ped: float,
    edge_basis_1: np.ndarray,
    edge_basis_2: np.ndarray,
    edge_integral_1: float,
    edge_integral_2: float,
    separatrix_to_pedestal_ratio: float,
) -> np.ndarray:
    """Construct a density profile with a requested center-to-pedestal ratio."""
    rho_core = rho[rho <= rho_ped]
    rho_edge = rho[rho >= rho_ped]
    gradient = float(np.log(peak_to_pedestal) / rho_ped)
    core_integral = _calc_jch_core_integral(gradient, rho_core, rho_ped)
    pedestal_value = volume_average / (core_integral + edge_integral_1 + separatrix_to_pedestal_ratio * edge_integral_2)

    profile = np.empty_like(rho)
    profile[rho <= rho_ped] = pedestal_value * np.exp(gradient * (rho_ped - rho_core))
    if rho_edge.size:
        profile[rho >= rho_ped] = pedestal_value * edge_basis_1 + (pedestal_value * separatrix_to_pedestal_ratio) * edge_basis_2

    return profile


def _build_jch_temperature_profile(
    volume_average: float,
    peak_to_pedestal: float,
    rho: np.ndarray,
    rho_ped: float,
    edge_basis_1: np.ndarray,
    edge_basis_2: np.ndarray,
    edge_integral_1: float,
    edge_integral_2: float,
    separatrix_temperature: float,
) -> np.ndarray:
    """Construct a temperature profile with a requested center-to-pedestal ratio."""
    rho_core = rho[rho <= rho_ped]
    rho_edge = rho[rho >= rho_ped]
    gradient = float(np.log(peak_to_pedestal) / rho_ped)
    pedestal_temperature = _calc_jch_temperature_pedestal_temperature(
        volume_average=volume_average,
        peak_to_pedestal=peak_to_pedestal,
        rho_core=rho_core,
        rho_ped=rho_ped,
        edge_integral_1=edge_integral_1,
        edge_integral_2=edge_integral_2,
        separatrix_temperature=separatrix_temperature,
    )

    if pedestal_temperature < separatrix_temperature:
        raise ValueError("Requested JCH temperature profile gives an unphysical pedestal below the separatrix temperature.")

    profile = np.empty_like(rho)
    profile[rho <= rho_ped] = pedestal_temperature * np.exp(gradient * (rho_ped - rho_core))
    if rho_edge.size:
        profile[rho >= rho_ped] = pedestal_temperature * edge_basis_1 + separatrix_temperature * edge_basis_2

    return profile


def calc_jch_profiles(
    average_electron_density: float,
    average_electron_temp: float,
    average_ion_temp: float,
    electron_density_volume_peaking: float | None,
    ion_density_volume_peaking: float | None,
    electron_temp_volume_peaking: float | None,
    ion_temp_volume_peaking: float | None,
    dilution: float,
    n_points: int,
    pedestal_width: float,
    t_sep: float,
    n_sep_ratio: float,
    rho: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Estimate JCH profiles with an exponential core and linear pedestal handoff.

    The public peaking inputs remain center-to-volume-average ratios. This
    helper converts them to the peak-to-pedestal ratios required by the JCH
    profile parameterization.
    """
    n_points = int(n_points)
    pedestal_width = float(pedestal_width)
    separatrix_temperature = float(t_sep)
    separatrix_to_pedestal_ratio = float(n_sep_ratio)

    if n_points < 3:
        raise ValueError("JCH profiles require at least three radial grid points.")
    if not 0.0 < pedestal_width < 1.0:
        raise ValueError("pedestal_width must lie strictly between 0 and 1 for JCH profiles.")
    if (electron_density_volume_peaking is None) != (ion_density_volume_peaking is None):
        raise ValueError("JCH density profiles require both electron and ion volume peaking values.")
    if (electron_temp_volume_peaking is None) != (ion_temp_volume_peaking is None):
        raise ValueError("JCH temperature profiles require both electron and ion volume peaking values.")

    rho_ped = 1.0 - pedestal_width
    if rho is None:
        rho = _build_profile_grid(n_points, rho_ped)
    else:
        rho = np.asarray(rho, dtype=float).copy()
        rho[_find_nearest_interior_grid_index(rho, rho_ped)] = rho_ped

    rho_core = rho[rho <= rho_ped]
    edge_basis_1, edge_basis_2, edge_integral_1, edge_integral_2 = _calc_jch_edge_integrals(rho, rho_ped, pedestal_width)

    common_density_profile_kwargs = dict(
        rho=rho,
        rho_ped=rho_ped,
        edge_basis_1=edge_basis_1,
        edge_basis_2=edge_basis_2,
        edge_integral_1=edge_integral_1,
        edge_integral_2=edge_integral_2,
        separatrix_to_pedestal_ratio=separatrix_to_pedestal_ratio,
    )
    common_temperature_profile_kwargs = dict(
        rho=rho,
        rho_ped=rho_ped,
        edge_basis_1=edge_basis_1,
        edge_basis_2=edge_basis_2,
        edge_integral_1=edge_integral_1,
        edge_integral_2=edge_integral_2,
        separatrix_temperature=separatrix_temperature,
    )

    electron_density_profile = None
    fuel_ion_density_profile = None
    if electron_density_volume_peaking is not None:
        electron_density_pedestal_peaking = _solve_jch_density_pedestal_peaking(
            target_volume_peaking=float(electron_density_volume_peaking),
            rho_core=rho_core,
            rho_ped=rho_ped,
            edge_integral_1=edge_integral_1,
            edge_integral_2=edge_integral_2,
            separatrix_to_pedestal_ratio=separatrix_to_pedestal_ratio,
        )
        ion_density_pedestal_peaking = _solve_jch_density_pedestal_peaking(
            target_volume_peaking=float(ion_density_volume_peaking),
            rho_core=rho_core,
            rho_ped=rho_ped,
            edge_integral_1=edge_integral_1,
            edge_integral_2=edge_integral_2,
            separatrix_to_pedestal_ratio=separatrix_to_pedestal_ratio,
        )
        electron_density_profile = _build_jch_density_profile(
            volume_average=float(average_electron_density),
            peak_to_pedestal=electron_density_pedestal_peaking,
            **common_density_profile_kwargs,
        )
        fuel_ion_density_profile = _build_jch_density_profile(
            volume_average=float(average_electron_density * dilution),
            peak_to_pedestal=ion_density_pedestal_peaking,
            **common_density_profile_kwargs,
        )

    electron_temp_profile = None
    ion_temp_profile = None
    if electron_temp_volume_peaking is not None:
        electron_temp_pedestal_peaking = _solve_jch_temperature_pedestal_peaking(
            target_volume_peaking=float(electron_temp_volume_peaking),
            volume_average=float(average_electron_temp),
            rho_core=rho_core,
            rho_ped=rho_ped,
            edge_integral_1=edge_integral_1,
            edge_integral_2=edge_integral_2,
            separatrix_temperature=separatrix_temperature,
        )
        ion_temp_pedestal_peaking = _solve_jch_temperature_pedestal_peaking(
            target_volume_peaking=float(ion_temp_volume_peaking),
            volume_average=float(average_ion_temp),
            rho_core=rho_core,
            rho_ped=rho_ped,
            edge_integral_1=edge_integral_1,
            edge_integral_2=edge_integral_2,
            separatrix_temperature=separatrix_temperature,
        )
        electron_temp_profile = _build_jch_temperature_profile(
            volume_average=float(average_electron_temp),
            peak_to_pedestal=electron_temp_pedestal_peaking,
            **common_temperature_profile_kwargs,
        )
        ion_temp_profile = _build_jch_temperature_profile(
            volume_average=float(average_ion_temp),
            peak_to_pedestal=ion_temp_pedestal_peaking,
            **common_temperature_profile_kwargs,
        )

    return rho, electron_density_profile, fuel_ion_density_profile, electron_temp_profile, ion_temp_profile
