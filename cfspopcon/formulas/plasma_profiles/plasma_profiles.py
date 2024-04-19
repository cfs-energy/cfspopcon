"""Estimate 1D plasma profiles of density and temperature."""

from typing import Optional

import numpy as np
from numpy import float64
from numpy.typing import NDArray

from ...algorithm_class import Algorithm
from ...named_options import ProfileForm
from ...unit_handling import Unitfull, ureg, wraps_ufunc
from .density_peaking import calc_density_peaking, calc_effective_collisionality
from .numerical_profile_fits import evaluate_density_and_temperature_profile_fits


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
    profile_form: ProfileForm,
    average_electron_density: Unitfull,
    line_averaged_electron_density: Unitfull,
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
) -> tuple[Unitfull, ...]:
    """Calculate density peaking and the corresponding density and temperature profiles.

    Args:
        profile_form: :term:`glossary link<profile_form>`
        average_electron_density: :term:`glossary link<average_electron_density>`
        line_averaged_electron_density: :term:`glossary link<line_averaged_electron_density>`
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

    Returns:
    `effective_collisionality`, :term:`ion_density_peaking`, :term:`electron_density_peaking`, :term:`peak_electron_density`, :term:`peak_electron_temp`, :term:`peak_ion_temp`, :term:`rho`, :term:`electron_density_profile`, :term:`fuel_ion_density_profile`, :term:`electron_temp_profile`, :term:`ion_temp_profile`

    """
    effective_collisionality = calc_effective_collisionality(
        line_averaged_electron_density, average_electron_temp, major_radius, z_effective
    )
    ion_density_peaking = calc_density_peaking(effective_collisionality, beta_toroidal, nu_noffset=ion_density_peaking_offset)
    electron_density_peaking = calc_density_peaking(effective_collisionality, beta_toroidal, nu_noffset=electron_density_peaking_offset)

    peak_electron_density = line_averaged_electron_density * electron_density_peaking
    peak_fuel_ion_density = line_averaged_electron_density * dilution * ion_density_peaking
    peak_electron_temp = average_electron_temp * temperature_peaking
    peak_ion_temp = average_ion_temp * temperature_peaking

    # Calculate the total fusion power by estimating density and temperature profiles and
    # using this to calculate fusion power profiles.
    (rho, electron_density_profile, fuel_ion_density_profile, electron_temp_profile, ion_temp_profile,) = calc_1D_plasma_profiles(
        profile_form,
        average_electron_density,
        line_averaged_electron_density,
        average_electron_temp,
        average_ion_temp,
        electron_density_peaking,
        ion_density_peaking,
        temperature_peaking,
        dilution,
        normalized_inverse_temp_scale_length,
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
        profile_form=None,
        average_electron_density=ureg.n19,
        line_averaged_electron_density=ureg.n19,
        average_electron_temp=ureg.keV,
        average_ion_temp=ureg.keV,
        electron_density_peaking=ureg.dimensionless,
        ion_density_peaking=ureg.dimensionless,
        temperature_peaking=ureg.dimensionless,
        dilution=ureg.dimensionless,
        normalized_inverse_temp_scale_length=ureg.dimensionless,
        npoints=None,
    ),
    output_core_dims=[("dim_rho",), ("dim_rho",), ("dim_rho",), ("dim_rho",), ("dim_rho",)],
)
def calc_1D_plasma_profiles(
    profile_form: ProfileForm,
    average_electron_density: float,
    line_averaged_electron_density: float,
    average_electron_temp: float,
    average_ion_temp: float,
    electron_density_peaking: float,
    ion_density_peaking: float,
    temperature_peaking: float,
    dilution: float,
    normalized_inverse_temp_scale_length: Optional[float] = None,
    npoints: int = 50,
) -> tuple[NDArray[float64], NDArray[float64], NDArray[float64], NDArray[float64], NDArray[float64]]:
    """Estimate density and temperature profiles.

    Args:
        profile_form: select between analytic fit or profiles from Pablo Rodriguez-Fernandez
        average_electron_density: [1e19 m^-3] :term:`glossary link<average_electron_density>`
        line_averaged_electron_density: [1e19 m^-3] :term:`glossary link<line_averaged_electron_density>`
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

    Raises:
        ValueError: if profile_form == prf and normalized_inverse_temp_scale_length is not set
    """
    if profile_form == ProfileForm.analytic:
        rho, electron_density_profile, fuel_ion_density_profile, electron_temp_profile, ion_temp_profile = calc_analytic_profiles(
            line_averaged_electron_density,
            average_electron_temp,
            average_ion_temp,
            electron_density_peaking,
            ion_density_peaking,
            temperature_peaking,
            dilution,
            npoints=npoints,
        )

    elif profile_form == ProfileForm.prf:
        if normalized_inverse_temp_scale_length is None:
            raise ValueError("normalized_inverse_temp_scale_length must be set if using profile_form = prf")

        rho, electron_density_profile, fuel_ion_density_profile, electron_temp_profile, ion_temp_profile = calc_prf_profiles(
            average_electron_density,
            average_electron_temp,
            average_ion_temp,
            electron_density_peaking,
            ion_density_peaking,
            temperature_peaking,
            dilution,
            normalized_inverse_temp_scale_length,
            npoints=npoints,
        )

    return rho, electron_density_profile, fuel_ion_density_profile, electron_temp_profile, ion_temp_profile


def calc_analytic_profiles(
    line_averaged_electron_density: float,
    average_electron_temp: float,
    average_ion_temp: float,
    electron_density_peaking: float,
    ion_density_peaking: float,
    temperature_peaking: float,
    dilution: float,
    npoints: int = 50,
) -> tuple[NDArray[float64], NDArray[float64], NDArray[float64], NDArray[float64], NDArray[float64]]:
    """Estimate density and temperature profiles using a simple analytic fit.

    Args:
        line_averaged_electron_density: [1e19 m^-3] :term:`glossary link<line_averaged_electron_density>`
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

    electron_density_profile = (
        line_averaged_electron_density * electron_density_peaking * ((1.0 - rho**2.0) ** (electron_density_peaking - 1.0))
    )
    fuel_ion_density_profile = (
        line_averaged_electron_density * dilution * (ion_density_peaking) * ((1.0 - rho**2.0) ** (ion_density_peaking - 1.0))
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
) -> tuple[NDArray[float64], NDArray[float64], NDArray[float64], NDArray[float64], NDArray[float64]]:
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
    rho = np.linspace(0, 1, num=npoints, endpoint=False)

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
