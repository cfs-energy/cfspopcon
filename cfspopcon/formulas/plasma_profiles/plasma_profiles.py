"""Estimate 1D plasma profiles of density and temperature."""

from bisect import bisect_left
from collections.abc import Callable
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq
from ...algorithm_class import Algorithm
from ...named_options import ProfileForm
from ...unit_handling import Unitfull, ureg, wraps_ufunc
from .density_peaking import calc_density_peaking, calc_effective_collisionality
from .numerical_profile_fits import evaluate_density_and_temperature_profile_fits

FloatArray: TypeAlias = NDArray[np.float64]
ProfileFamily: TypeAlias = tuple[FloatArray, FloatArray | None, FloatArray | None, FloatArray | None, FloatArray | None]


@Algorithm.register_algorithm(
    return_keys=[
        "effective_collisionality",
        "ion_density_peaking",
        "electron_density_peaking",
        "peak_electron_density",
        "peak_fuel_ion_density",
        "peak_electron_temp",
        "peak_ion_temp",
        "rho",
        "electron_density_profile",
        "fuel_ion_density_profile",
        "electron_temp_profile",
        "ion_temp_profile",
        "ion_density_pedestal_peaking",
        "electron_density_pedestal_peaking",
        "electron_temp_pedestal_peaking",
        "ion_temp_pedestal_peaking",
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
        :term:`peak_electron_density`, :term:`peak_fuel_ion_density`, :term:`peak_electron_temp`,
        :term:`peak_ion_temp`, :term:`rho`, :term:`electron_density_profile`, :term:`fuel_ion_density_profile`,
        :term:`electron_temp_profile`, :term:`ion_temp_profile`, :term:`ion_density_pedestal_peaking`,
        :term:`electron_density_pedestal_peaking`, :term:`electron_temp_pedestal_peaking`,
        :term:`ion_temp_pedestal_peaking`. The pedestal peaking outputs are only
        defined for JCH branches; other profile forms return ``NaN`` placeholders
        so the Algorithm output contract stays fixed.
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
        # Preserve the fixed return contract even when the chosen profile family
        # has no pedestal concept.
        electron_density_pedestal_peaking = electron_density_peaking * np.nan
        ion_density_pedestal_peaking = ion_density_peaking * np.nan
        electron_temp_pedestal_peaking = temperature_peaking * np.nan
        ion_temp_pedestal_peaking = temperature_peaking * np.nan

    return (
        effective_collisionality,
        ion_density_peaking,
        electron_density_peaking,
        peak_electron_density,
        peak_fuel_ion_density,
        peak_electron_temp,
        peak_ion_temp,
        rho,
        electron_density_profile,
        fuel_ion_density_profile,
        electron_temp_profile,
        ion_temp_profile,
        ion_density_pedestal_peaking,
        electron_density_pedestal_peaking,
        electron_temp_pedestal_peaking,
        ion_temp_pedestal_peaking,
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
    JCH parameterization. Different profile families may use different
    construction grids internally, but the returned profiles always share one
    common output ``rho`` grid. In mixed-form cases this means PRF can be built
    on its own default-compatible grid and then remapped onto a JCH-shaped
    output grid without letting JCH-only pedestal choices distort the PRF
    volume average.

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
            All profile grids stop about one tenth of a grid spacing inside the
            LCFS so hollow analytic profiles remain finite without letting the
            final trapezoid dominate the volume integral.
        pedestal_width: Pedestal width in normalized rho for JCH profiles.
        t_sep: Separatrix temperature used to anchor the JCH edge temperature profile.
        n_sep_ratio: Ratio of separatrix density to pedestal density for JCH profiles.

    Returns:
        :term:`rho` [~], :term:`electron_density_profile` [1e19 m^-3],
        :term:`fuel_ion_density_profile` [1e19 m^-3], :term:`electron_temp_profile` [keV],
        :term:`ion_temp_profile` [keV]
    """
    needs_jch_profiles = density_profile_form == ProfileForm.jch or temp_profile_form == ProfileForm.jch
    default_rho_grid = _build_profile_grid(n_points_for_confined_region_profiles)
    # When a JCH branch is requested, expose the edge-refined JCH grid as the
    # public rho coordinate. Other profile families are remapped onto it if
    # needed so downstream consumers still see one shared radial coordinate.
    rho_grid = _build_profile_grid(
        n_points_for_confined_region_profiles,
        (1.0 - float(pedestal_width)) if needs_jch_profiles else None,
    )
    selected_forms = {density_profile_form, temp_profile_form}
    family_profiles: dict[ProfileForm, ProfileFamily] = dict()

    # Only build the profile families that were actually requested; mixed-form
    # runs then combine the density branch from one family with the temperature
    # branch from another on the shared output grid.
    for profile_form in selected_forms:
        if profile_form == ProfileForm.analytic:
            family_profiles[profile_form] = calc_analytic_profiles(
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
        elif profile_form == ProfileForm.prf:
            prf_profiles = calc_prf_profiles(
                average_electron_density=average_electron_density,
                average_electron_temp=average_electron_temp,
                average_ion_temp=average_ion_temp,
                electron_density_peaking=electron_density_peaking,
                ion_density_peaking=ion_density_peaking,
                temperature_peaking=temperature_peaking,
                dilution=dilution,
                normalized_inverse_temp_scale_length=normalized_inverse_temp_scale_length,
                npoints=n_points_for_confined_region_profiles,
                rho=default_rho_grid,
            )

            (
                prf_rho,
                prf_electron_density_profile,
                prf_fuel_ion_density_profile,
                prf_electron_temp_profile,
                prf_ion_temp_profile,
            ) = prf_profiles
            # PRF uses its own construction grid and pedestal assumptions, so in
            # mixed-form cases we remap it onto the common output rho grid
            # rather than letting JCH-specific grid choices perturb the PRF
            # volume averages directly.
            prf_electron_density_profile = _remap_profile_onto_grid(
                prf_electron_density_profile,
                prf_rho,
                rho_grid,
                target_volume_average=float(average_electron_density),
            )
            prf_fuel_ion_density_profile = _remap_profile_onto_grid(
                prf_fuel_ion_density_profile,
                prf_rho,
                rho_grid,
                target_volume_average=float(average_electron_density * dilution),
            )
            prf_electron_temp_profile = _remap_profile_onto_grid(
                prf_electron_temp_profile,
                prf_rho,
                rho_grid,
                target_volume_average=float(average_electron_temp),
            )
            prf_ion_temp_profile = _remap_profile_onto_grid(
                prf_ion_temp_profile,
                prf_rho,
                rho_grid,
                target_volume_average=float(average_ion_temp),
            )
            prf_rho = rho_grid

            family_profiles[profile_form] = (
                prf_rho,
                prf_electron_density_profile,
                prf_fuel_ion_density_profile,
                prf_electron_temp_profile,
                prf_ion_temp_profile,
            )
        else:
            family_profiles[profile_form] = calc_jch_profiles(
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

    density_family = family_profiles[density_profile_form]
    temp_family = family_profiles[temp_profile_form]

    density_rho = density_family[0]
    electron_density_profile = density_family[1]
    fuel_ion_density_profile = density_family[2]
    temp_rho = temp_family[0]
    electron_temp_profile = temp_family[3]
    ion_temp_profile = temp_family[4]

    assert np.allclose(density_rho, rho_grid)
    assert np.allclose(temp_rho, rho_grid)
    assert electron_density_profile is not None
    assert fuel_ion_density_profile is not None
    assert electron_temp_profile is not None
    assert ion_temp_profile is not None

    return (
        rho_grid,
        electron_density_profile,
        fuel_ion_density_profile,
        electron_temp_profile,
        ion_temp_profile,
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
    """Convert volume peaking values into the JCH peak-to-pedestal ratios.

    The public API reports peaking as center-to-volume-average ratios for every
    profile family. JCH profile construction instead needs center-to-pedestal
    ratios, so this helper performs the inversion using the same pedestal
    geometry and integration rules as :func:`calc_jch_profiles`.

    Returns ``NaN`` for branches that are not using ``ProfileForm.jch``.
    """
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
        rho: optional output grid. If omitted, the helper builds its own
            regularized profile grid.

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
        rho: optional construction grid. The PRF fit tables were tuned for
            their own default grid, so mixed-form callers typically build PRF on
            that grid first and remap onto the shared output grid afterward.

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
    """Find the nearest point in a sorted grid using a bisection search.

    This is used when an externally supplied grid needs to be snapped to the
    exact JCH pedestal knee without scanning the whole array.
    """
    insertion_index = bisect_left(values.tolist(), target)

    if insertion_index == 0:
        return 0
    if insertion_index == len(values):
        return len(values) - 1
    if (values[insertion_index] - target) < (target - values[insertion_index - 1]):
        return insertion_index

    return insertion_index - 1


def _find_nearest_interior_grid_index(values: np.ndarray, target: float) -> int:
    """Find the nearest interior point in a sorted grid.

    JCH profile construction reserves the axis and edge points, so the pedestal
    knee must be snapped onto an interior point only.
    """
    if len(values) < 3:
        raise ValueError("JCH profiles require at least three radial grid points to preserve both the axis and separatrix.")

    return 1 + _find_nearest_grid_index(values[1:-1], target)


def _calc_profile_grid_edge_nudge(npoints: int) -> float:
    """Return the resolution-dependent LCFS offset used to regularize the grid.

    Hollow analytic profiles are singular exactly at ``rho = 1``. Rather than
    using one fixed epsilon, the nudge is scaled to the grid spacing so the last
    trapezoid stays well behaved across different resolutions.
    """
    if npoints <= 1:
        return 0.0

    # Choose the endpoint offset so it is one tenth of the induced grid
    # spacing: nudge = 0.1 * drho, drho = (1 - nudge) / (npoints - 1).
    return 0.1 / (npoints - 1 + 0.1)


def _build_profile_grid(npoints: int, rho_ped: float | None = None) -> np.ndarray:
    """Build the radial grid and optionally reserve four points across the pedestal.

    Non-JCH grids stop about one tenth of a grid spacing inside the LCFS so the
    analytic hollow-profile form is regularized without overweighting the final
    trapezoid. JCH grids use the same offset so mixed analytic/JCH calls can
    safely share a single grid without evaluating analytic hollow profiles at
    ``rho = 1``.
    """
    edge_nudge = _calc_profile_grid_edge_nudge(npoints)

    if rho_ped is None:
        return np.linspace(0.0, 1.0 - edge_nudge, num=npoints)

    # Reserve four samples for the pedestal region: the knee, two interior
    # points that can capture some curvature, and the edge point.
    pedestal_points = 4
    if npoints < pedestal_points + 1:
        raise ValueError("JCH profile grids require at least five radial points to preserve the axis and four pedestal samples.")

    core_points = npoints - pedestal_points + 1
    rho_core = np.linspace(0.0, rho_ped, num=core_points)
    rho_pedestal = np.linspace(rho_ped, 1.0 - edge_nudge, num=pedestal_points)
    return cast(FloatArray, np.concatenate((rho_core, rho_pedestal[1:])))


def _remap_profile_onto_grid(
    profile: np.ndarray,
    source_rho: np.ndarray,
    target_rho: np.ndarray,
    target_volume_average: float,
) -> np.ndarray:
    """Interpolate a profile onto a new rho grid and renormalize its volume average.

    Mixed-form runs keep one public ``rho`` output even when different profile
    families need different construction grids internally. This helper is what
    lets PRF keep its own native construction grid and then move onto the common
    output grid without losing the requested volume average.
    """
    if np.allclose(source_rho, target_rho):
        return profile

    remapped_profile = cast(FloatArray, np.interp(target_rho, source_rho, profile))
    if np.isclose(target_volume_average, 0.0):
        return np.zeros_like(remapped_profile)

    # Renormalize with the same cylindrical-volume measure used elsewhere in
    # the profile code so the remapped profile still hits the requested average.
    remapped_volume_average = float(np.trapezoid(remapped_profile * 2.0 * target_rho, x=target_rho))
    if np.isclose(remapped_volume_average, 0.0):
        raise ValueError("Cannot renormalize a remapped profile with zero volume average.")

    return remapped_profile * (target_volume_average / remapped_volume_average)


def _calc_jch_core_integral(gradient: float, rho_core: np.ndarray, rho_ped: float) -> float:
    """Integrate a pedestal-normalized exponential core profile over the confined volume."""
    profile = np.exp(gradient * (rho_ped - rho_core))
    return float(np.trapezoid(profile * 2.0 * rho_core, x=rho_core))


def _calc_jch_edge_integrals(rho: np.ndarray, rho_ped: float, pedestal_width: float) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Compute the linear pedestal basis functions and their volume integrals.

    The JCH edge is represented as a linear blend between the pedestal value and
    the separatrix anchor. Returning both basis arrays and their integrals lets
    the solver and profile builder share the same geometry bookkeeping.
    """
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
    """Return the peak-to-volume-average ratio for a JCH density profile.

    ``peak_to_pedestal`` is the natural JCH input. This helper translates it
    into the public volume-peaking definition used elsewhere in the codebase.
    """
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
    """Return the pedestal temperature implied by a JCH peak-to-pedestal ratio.

    Temperature profiles are additionally constrained by the separatrix
    temperature, so the pedestal value must be solved from the requested volume
    average before the profile can be constructed.
    """
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
    """Solve for the JCH peak-to-pedestal ratio that matches a target volume peaking.

    The mapping from peak-to-pedestal to peak-to-volume is monotonic but does
    not have a closed-form inverse here, so we bracket the valid range and use a
    scalar root find. Temperature profiles can optionally impose an additional
    upper bound from the separatrix-temperature constraint.
    """
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
        # First ensure the flat-core limit is already physical. If not, no valid
        # peak-to-pedestal ratio exists for the requested average temperature.
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
        # Find the largest peak-to-pedestal ratio that still leaves the
        # pedestal at or above the separatrix temperature.
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
    """Construct a density profile with a requested center-to-pedestal ratio.

    The pedestal value is solved so that the full piecewise profile integrates
    back to the requested volume average.
    """
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
    """Construct a temperature profile with a requested center-to-pedestal ratio.

    Unlike density, the edge branch is anchored to an absolute separatrix
    temperature, so the pedestal temperature must be solved before the profile
    can be filled in.
    """
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
    profile parameterization. Density and temperature branches can be requested
    independently; the unused branch returns ``None`` so the caller can splice
    together mixed-form runs.
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
        # Preserve the caller's overall grid while guaranteeing that the exact
        # pedestal knee appears on one interior point.
        rho[_find_nearest_interior_grid_index(rho, rho_ped)] = rho_ped

    rho_core = rho[rho <= rho_ped]
    edge_basis_1, edge_basis_2, edge_integral_1, edge_integral_2 = _calc_jch_edge_integrals(rho, rho_ped, pedestal_width)

    electron_density_profile = None
    fuel_ion_density_profile = None
    if electron_density_volume_peaking is not None:
        assert ion_density_volume_peaking is not None
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
            rho=rho,
            rho_ped=rho_ped,
            edge_basis_1=edge_basis_1,
            edge_basis_2=edge_basis_2,
            edge_integral_1=edge_integral_1,
            edge_integral_2=edge_integral_2,
            separatrix_to_pedestal_ratio=separatrix_to_pedestal_ratio,
        )
        fuel_ion_density_profile = _build_jch_density_profile(
            volume_average=float(average_electron_density * dilution),
            peak_to_pedestal=ion_density_pedestal_peaking,
            rho=rho,
            rho_ped=rho_ped,
            edge_basis_1=edge_basis_1,
            edge_basis_2=edge_basis_2,
            edge_integral_1=edge_integral_1,
            edge_integral_2=edge_integral_2,
            separatrix_to_pedestal_ratio=separatrix_to_pedestal_ratio,
        )

    electron_temp_profile = None
    ion_temp_profile = None
    if electron_temp_volume_peaking is not None:
        assert ion_temp_volume_peaking is not None
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
            rho=rho,
            rho_ped=rho_ped,
            edge_basis_1=edge_basis_1,
            edge_basis_2=edge_basis_2,
            edge_integral_1=edge_integral_1,
            edge_integral_2=edge_integral_2,
            separatrix_temperature=separatrix_temperature,
        )
        ion_temp_profile = _build_jch_temperature_profile(
            volume_average=float(average_ion_temp),
            peak_to_pedestal=ion_temp_pedestal_peaking,
            rho=rho,
            rho_ped=rho_ped,
            edge_basis_1=edge_basis_1,
            edge_basis_2=edge_basis_2,
            edge_integral_1=edge_integral_1,
            edge_integral_2=edge_integral_2,
            separatrix_temperature=separatrix_temperature,
        )

    return rho, electron_density_profile, fuel_ion_density_profile, electron_temp_profile, ion_temp_profile
