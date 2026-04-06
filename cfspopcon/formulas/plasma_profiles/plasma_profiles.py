"""Estimate 1D plasma profiles of density and temperature."""

from typing import Any

import numpy as np

from ...algorithm_class import Algorithm
from ...named_options import ProfileForm
from ...unit_handling import Unitfull, ureg, wraps_ufunc
from .density_peaking import calc_density_peaking, calc_effective_collisionality
from .numerical_profile_fits import evaluate_density_and_temperature_profile_fits
from scipy.optimize import root_scalar
from cfspopcon.unit_handling import magnitude


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

    Returns:
    `effective_collisionality`, :term:`ion_density_peaking`, :term:`electron_density_peaking`, :term:`peak_electron_density`, :term:`peak_electron_temp`, :term:`peak_ion_temp`, :term:`rho`, :term:`electron_density_profile`, :term:`fuel_ion_density_profile`, :term:`electron_temp_profile`, :term:`ion_temp_profile`

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

    Args:
        density_profile_form: :term:`<glossary link<density_profile_form>`
        temp_profile_form: :term:`<glossary link<temp_profile_form>`
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
        :term:`rho` [~], :term:`electron_density_profile` [1e19 m^-3], fuel_ion_density_profile [1e19 m^-3], :term:`electron_temp_profile` [keV], :term:`ion_temp_profile` [keV]
    """
    kwargs: dict[str, Any] = dict(
        average_electron_density=average_electron_density,
        average_electron_temp=average_electron_temp,
        average_ion_temp=average_ion_temp,
        electron_density_peaking=electron_density_peaking,
        ion_density_peaking=ion_density_peaking,
        temperature_peaking=temperature_peaking,
        dilution=dilution,
        npoints=n_points_for_confined_region_profiles,
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

    (
        rho_3,
        electron_density_profiles[ProfileForm.jch],
        fuel_ion_density_profiles[ProfileForm.jch],
        electron_temp_profiles[ProfileForm.jch],
        ion_temp_profiles[ProfileForm.jch],
    ) = calc_jch_profiles(
        average_electron_density=average_electron_density,
        average_electron_temp=average_electron_temp,
        average_ion_temp=average_ion_temp,
        electron_density_peaking=electron_density_peaking,
        temperature_peaking=temperature_peaking,
        dilution=dilution,
        n_points=n_points_for_confined_region_profiles,
        pedestal_width=pedestal_width,
        t_sep=t_sep,
        n_sep_ratio=n_sep_ratio,
    )

    assert np.allclose(rho_1, rho_2)

    return (
        rho_1,
        electron_density_profiles[density_profile_form],
        fuel_ion_density_profiles[density_profile_form],
        electron_temp_profiles[temp_profile_form],
        ion_temp_profiles[temp_profile_form],
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
    rho = np.linspace(0, 1, num=npoints, endpoint=False)

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
    rho: np.ndarray = np.linspace(0.0, 1.0, num=npoints, endpoint=False)

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


def _safe_extract_numpy(val):
    if val is None:
        return None
    try:
        mag = magnitude(val)
        return np.asarray(mag.values if hasattr(mag, "values") else mag, dtype=float)
    except Exception:
        return np.asarray(val.values if hasattr(val, "values") else val, dtype=float)


def _solve_jch_density_gradient(target_peaking, rho, rho_ped, I_edge1, I_edge2, n_sep_r):
    def objective(a_L):
        core_mask = rho <= rho_ped
        prof_core = np.exp(a_L * (rho_ped - rho[core_mask]))
        I_core = np.trapz(prof_core * 2.0 * rho[core_mask], x=rho[core_mask])
        vol_avg = I_core + I_edge1 + n_sep_r * I_edge2
        return (prof_core[0] / vol_avg) - target_peaking

    res = root_scalar(objective, bracket=[-2.0, 20.0], method="brentq")
    a_L = res.root

    core_mask = rho <= rho_ped
    prof_core = np.exp(a_L * (rho_ped - rho[core_mask]))
    I_core = np.trapz(prof_core * 2.0 * rho[core_mask], x=rho[core_mask])
    return a_L, I_core


def calc_jch_profiles(
    average_electron_density,
    average_electron_temp,
    average_ion_temp,
    electron_density_peaking,
    temperature_peaking,
    dilution,
    n_points,
    pedestal_width,
    t_sep,
    n_sep_ratio,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # 1. Extract inputs
    ne = _safe_extract_numpy(average_electron_density)
    te = _safe_extract_numpy(average_electron_temp)
    ti = _safe_extract_numpy(average_ion_temp)
    dil = _safe_extract_numpy(dilution)
    t_sep_arr = _safe_extract_numpy(t_sep)

    # These shape the profile geometry, so we extract them as scalars
    w_ped = float(np.mean(_safe_extract_numpy(pedestal_width)))
    n_sep_r = float(np.mean(_safe_extract_numpy(n_sep_ratio)))
    n_peak = float(np.mean(_safe_extract_numpy(electron_density_peaking)))
    t_peak = float(np.mean(_safe_extract_numpy(temperature_peaking)))

    # 2. Generate 1D Rho array
    rho_ped = 1.0 - w_ped
    rho = np.linspace(0.0, 1.0, int(n_points))

    # 3. Pre-calculate Edge Basis Integrals
    edge_mask = rho > rho_ped
    if np.any(edge_mask):
        rho_edge = rho[edge_mask]
        basis1 = (1.0 - rho_edge) / w_ped
        basis2 = (rho_edge - rho_ped) / w_ped
        I_edge1 = np.trapz(basis1 * 2.0 * rho_edge, x=rho_edge)
        I_edge2 = np.trapz(basis2 * 2.0 * rho_edge, x=rho_edge)
    else:
        I_edge1, I_edge2 = 0.0, 0.0

    # 4. Solve Gradients
    val_a_Ln, I_core_n = _solve_jch_density_gradient(n_peak, rho, rho_ped, I_edge1, I_edge2, n_sep_r)

    val_a_LT = np.log(t_peak) / rho_ped
    core_mask = rho <= rho_ped
    prof_core_T = np.exp(val_a_LT * (rho_ped - rho[core_mask]))
    I_core_T = np.trapz(prof_core_T * 2.0 * rho[core_mask], x=rho[core_mask])

    # 5. Exact Linear Handoffs (Zero division hazards here)
    nped = ne / (I_core_n + I_edge1 + n_sep_r * I_edge2)

    tped = (te - (t_sep_arr * I_edge2)) / (I_core_T + I_edge1)
    tped = np.maximum(tped, t_sep_arr)

    # Safely calculate ion pedestal independently instead of using ratios
    tiped = (ti - (t_sep_arr * I_edge2)) / (I_core_T + I_edge1)
    tiped = np.maximum(tiped, t_sep_arr)

    # 6. Construct Full Profiles
    n_prof = np.zeros_like(ne * rho)
    t_prof = np.zeros_like(te * rho)
    ti_prof = np.zeros_like(ti * rho)

    # Core
    n_prof[..., core_mask] = nped * np.exp(val_a_Ln * (rho_ped - rho[core_mask]))
    t_prof[..., core_mask] = tped * np.exp(val_a_LT * (rho_ped - rho[core_mask]))
    ti_prof[..., core_mask] = tiped * np.exp(val_a_LT * (rho_ped - rho[core_mask]))

    # Edge
    if np.any(edge_mask):
        n_prof[..., edge_mask] = nped * basis1 + (nped * n_sep_r) * basis2
        t_prof[..., edge_mask] = tped * basis1 + t_sep_arr * basis2
        ti_prof[..., edge_mask] = tiped * basis1 + t_sep_arr * basis2

    ni_prof = n_prof * dil

    return rho, n_prof, ni_prof, t_prof, ti_prof


@Algorithm.register_algorithm(return_keys=["P_in", "P_auxiliary_absorbed", "energy_confinement_time", "required_H98"])
@wraps_ufunc(
    return_units=dict(
        P_in=ureg.MW,
        P_auxiliary_absorbed=ureg.MW,
        energy_confinement_time=ureg.s,
        required_H98=ureg.dimensionless,
    ),
    input_units={
        "plasma_stored_energy": ureg.MJ,
        "P_ohmic": ureg.MW,
        "P_fusion": ureg.MW,
        "average_electron_density": ureg.n19,
        "plasma_current": ureg.MA,
        "major_radius": ureg.m,
        "minor_radius": ureg.m,
        "magnetic_field_on_axis": ureg.T,
        "average_ion_mass": ureg.amu,
        "areal_elongation": ureg.dimensionless,
        "P_auxiliary_launched": ureg.MW,
        "fraction_of_external_power_coupled": ureg.dimensionless,
    },
    output_core_dims=[(), (), (), ()],
)
def calc_power_balance_from_input_P_aux(
    plasma_stored_energy,
    P_ohmic,
    P_fusion,
    average_electron_density,
    plasma_current,
    major_radius,
    minor_radius,
    magnetic_field_on_axis,
    average_ion_mass,
    areal_elongation,
    P_auxiliary_launched,
    fraction_of_external_power_coupled,
):
    # 1. Calculate absorbed auxiliary power from the input
    P_aux_abs = P_auxiliary_launched * fraction_of_external_power_coupled

    # 2. Total power loss required to maintain steady state
    P_alpha = 0.2 * P_fusion
    P_in = P_alpha + P_ohmic + P_aux_abs

    # 3. Required confinement time
    tau_req = plasma_stored_energy / np.maximum(P_in, 1e-3)

    # 4. ITER98y2 Scaling Prediction based on the ACTUAL total heating
    epsilon = minor_radius / major_radius
    tau_scaling = (
        0.0562
        * (plasma_current**0.93)
        * (magnetic_field_on_axis**0.15)
        * (np.maximum(P_in, 1e-3) ** -0.69)
        * (average_electron_density**0.41)
        * (average_ion_mass**0.19)
        * (major_radius**1.97)
        * (epsilon**0.58)
        * (areal_elongation**0.78)
    )

    # 5. Required H98 factor
    required_H98 = tau_req / tau_scaling

    return P_in, P_aux_abs, tau_req, required_H98
