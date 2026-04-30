"""Estimate 1D plasma profiles of density and temperature."""

from typing import cast

import numpy as np
from numpy.typing import NDArray

from ...algorithm_class import Algorithm
from ...helpers import get_item
from ...named_options import ProfileForm
from ...unit_handling import Unitfull, ureg, wraps_ufunc
from .density_peaking import calc_density_peaking, calc_effective_collisionality
from .numerical_profile_fits import evaluate_density_and_temperature_profile_fits

FloatArray = NDArray[np.float64]
ProfileFamily = tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]


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

    Returns:
    `effective_collisionality`, :term:`ion_density_peaking`, :term:`electron_density_peaking`, 
    :term:`peak_electron_density`, :term:`peak_fuel_ion_density`, :term:`peak_electron_temp`, 
    :term:`peak_ion_temp`, :term:`rho`, :term:`electron_density_profile`, 
    :term:`fuel_ion_density_profile`, :term:`electron_temp_profile`, :term:`ion_temp_profile`

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
    )

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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate density and temperature profiles.

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
        n_points_for_confined_region_profiles: number of points to return in profile

    Returns:
        :term:`rho` [~], :term:`electron_density_profile` [1e19 m^-3],
        :term:`fuel_ion_density_profile` [1e19 m^-3], :term:`electron_temp_profile` [keV],
        :term:`ion_temp_profile` [keV]
    """
    rho_grid = _build_profile_grid(n_points_for_confined_region_profiles)
    selected_forms = {density_profile_form, temp_profile_form}
    family_profiles: dict[ProfileForm, ProfileFamily] = dict()

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
            # PRF fits are tuned for their own default grid.
            default_rho = _build_profile_grid(n_points_for_confined_region_profiles)
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
                rho=default_rho,
            )

            (
                prf_rho,
                prf_ne,
                prf_ni,
                prf_te,
                prf_ti,
            ) = prf_profiles

            # Remap PRF onto the shared grid while preserving volume averages.
            family_profiles[profile_form] = (
                rho_grid,
                _remap_profile_onto_grid(prf_ne, prf_rho, rho_grid, float(average_electron_density)),
                _remap_profile_onto_grid(prf_ni, prf_rho, rho_grid, float(average_electron_density * dilution)),
                _remap_profile_onto_grid(prf_te, prf_rho, rho_grid, float(average_electron_temp)),
                _remap_profile_onto_grid(prf_ti, prf_rho, rho_grid, float(average_ion_temp)),
            )

    density_family = family_profiles[density_profile_form]
    temp_family = family_profiles[temp_profile_form]

    return (
        rho_grid,
        density_family[1],
        density_family[2],
        temp_family[3],
        temp_family[4],
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
) -> ProfileFamily:
    """Estimate density and temperature profiles using a simple analytic fit."""
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
) -> ProfileFamily:
    """Estimate density and temperature profiles using profiles from Pablo Rodriguez-Fernandez."""
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


def _build_profile_grid(npoints: int) -> np.ndarray:
    """Build the radial grid with a small LCFS offset."""
    return np.linspace(0.0, 1.0, num=npoints, endpoint=False)


def _remap_profile_onto_grid(
    profile: np.ndarray,
    source_rho: np.ndarray,
    target_rho: np.ndarray,
    target_volume_average: float,
) -> np.ndarray:
    """Interpolate a profile and renormalize its volume average."""
    if np.allclose(source_rho, target_rho):
        return profile

    remapped_profile = cast("FloatArray", np.interp(target_rho, source_rho, profile))
    if np.isclose(target_volume_average, 0.0):
        return np.zeros_like(remapped_profile)

    remapped_volume_average = float(np.trapezoid(remapped_profile * 2.0 * target_rho, x=target_rho))
    if np.isclose(remapped_volume_average, 0.0):
        return remapped_profile

    return remapped_profile * (target_volume_average / remapped_volume_average)