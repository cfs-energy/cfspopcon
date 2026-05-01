"""Estimate 1D plasma profiles of density and temperature."""


import numpy as np
import xarray as xr

from ...algorithm_class import Algorithm, CompositeAlgorithm
from .numerical_profile_fits import evaluate_density_and_temperature_profile_fits


@Algorithm.register_algorithm(
    return_keys=["rho"]
)
def define_radial_grid(n_points_for_confined_region_profiles: int = 50) -> np.ndarray:
    """Define the radial grid for profiles."""
    x = np.linspace(0, 1, num=n_points_for_confined_region_profiles, endpoint=False)
    return xr.DataArray(x, coords=dict(dim_rho=x))

@Algorithm.register_algorithm(return_keys=["electron_density_profile", "fuel_ion_density_profile", "electron_temp_profile", "ion_temp_profile"])
def calc_analytic_profiles(
    rho: np.ndarray,
    average_electron_density: float,
    average_electron_temp: float,
    average_ion_temp: float,
    electron_density_peaking: float,
    ion_density_peaking: float,
    temperature_peaking: float,
    dilution: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate density and temperature profiles using a simple analytic fit.

    Args:
        rho : [~] :term:`glossary link<rho>`
        average_electron_density: [1e19 m^-3] :term:`glossary link<average_electron_density>`
        average_electron_temp: [keV] :term:`glossary link<average_electron_temp>`
        average_ion_temp: [keV] :term:`glossary link<average_ion_temp>`
        electron_density_peaking: [~] :term:`glossary link<electron_density_peaking>`
        ion_density_peaking: [~] :term:`glossary link<ion_density_peaking>`
        temperature_peaking: [~] :term:`glossary link<temperature_peaking>`
        dilution: dilution of main ions [~]

    Returns:
        :term:`electron_density_profile` [1e19 m^-3], fuel_ion_density_profile [1e19 m^-3], :term:`electron_temp_profile` [keV], :term:`ion_temp_profile` [keV]
    """
    electron_density_profile = average_electron_density * electron_density_peaking * ((1.0 - rho**2.0) ** (electron_density_peaking - 1.0))
    fuel_ion_density_profile = (
        average_electron_density * dilution * (ion_density_peaking) * ((1.0 - rho**2.0) ** (ion_density_peaking - 1.0))
    )
    electron_temp_profile = average_electron_temp * temperature_peaking * ((1.0 - rho**2.0) ** (temperature_peaking - 1.0))
    ion_temp_profile = average_ion_temp * temperature_peaking * ((1.0 - rho**2.0) ** (temperature_peaking - 1.0))

    return electron_density_profile, fuel_ion_density_profile, electron_temp_profile, ion_temp_profile

@Algorithm.register_algorithm(return_keys=["electron_density_profile", "fuel_ion_density_profile", "electron_temp_profile", "ion_temp_profile"])
def calc_prf_profiles(
    rho: np.ndarray,
    average_electron_density: float,
    average_electron_temp: float,
    average_ion_temp: float,
    electron_density_peaking: float,
    ion_density_peaking: float,
    temperature_peaking: float,
    dilution: float,
    normalized_inverse_temp_scale_length: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate density and temperature profiles using profiles from Pablo Rodriguez-Fernandez.

    Args:
        rho: [~] :term:`glossary link<rho>`
        average_electron_density: [1e19 m^-3] :term:`glossary link<average_electron_density>`
        average_electron_temp: [keV] :term:`glossary link<average_electron_temp>`
        average_ion_temp: [keV] :term:`glossary link<average_ion_temp>`
        electron_density_peaking: [~] :term:`glossary link<electron_density_peaking>`
        ion_density_peaking: [~] :term:`glossary link<ion_density_peaking>`
        temperature_peaking: [~] :term:`glossary link<temperature_peaking>`
        dilution: dilution of main ions [~]
        normalized_inverse_temp_scale_length: [~] :term:`glossary link<normalized_inverse_temp_scale_length>`

    Returns:
        :term:`electron_density_profile` [1e19 m^-3], fuel_ion_density_profile [1e19 m^-3], :term:`electron_temp_profile` [keV], :term:`ion_temp_profile` [keV]
    """
    electron_temp_profile, electron_density_profile = evaluate_density_and_temperature_profile_fits(
        rho=rho,
        T_avol = average_electron_temp,
        n_avol = average_electron_density,
        temperature_peaking = temperature_peaking,
        nu_n = electron_density_peaking,
        aLT=normalized_inverse_temp_scale_length,
        dataset="PRF",
    )
    ion_temp_profile, fuel_ion_density_profile = evaluate_density_and_temperature_profile_fits(
        rho=rho,
        T_avol = average_ion_temp,
        n_avol = average_electron_density * dilution,
        temperature_peaking = temperature_peaking,
        nu_n = ion_density_peaking,
        aLT=normalized_inverse_temp_scale_length,
        dataset="PRF",
    )

    return electron_density_profile, fuel_ion_density_profile, electron_temp_profile, ion_temp_profile


calc_peak_electron_temp = Algorithm.from_single_function(
    lambda average_electron_temp, temperature_peaking: average_electron_temp * temperature_peaking,
    return_keys = ["peak_electron_temp"],
    name = "calc_peak_electron_temp"
)
calc_peak_ion_temp = Algorithm.from_single_function(
    lambda average_ion_temp, temperature_peaking: average_ion_temp * temperature_peaking,
    return_keys = ["peak_ion_temp"],
    name = "calc_peak_ion_temp"
)

calc_peaking_and_analytic_profiles = CompositeAlgorithm(
    algorithms=[
        Algorithm.get_algorithm(alg)
        for alg in [
            "calc_effective_collisionality",
            "calc_ion_density_peaking",
            "calc_electron_density_peaking",
            "calc_peak_electron_temp",
            "calc_peak_ion_temp",
            "define_radial_grid",
            "calc_analytic_profiles",
        ]
    ],
    name="calc_peaking_and_analytic_profiles",
    register=True,
)

calc_peaking_and_prf_profiles = CompositeAlgorithm(
    algorithms=[
        Algorithm.get_algorithm(alg)
        for alg in [
            "calc_effective_collisionality",
            "calc_ion_density_peaking",
            "calc_electron_density_peaking",
            "calc_peak_electron_temp",
            "calc_peak_ion_temp",
            "define_radial_grid",
            "calc_prf_profiles",
        ]
    ],
    name="calc_peaking_and_prf_profiles",
    register=True,
)

@Algorithm.register_algorithm(return_keys=[])
def calc_peaked_profiles():
    """Deprecated entry point for setting up profiles.""" #TODO: remove in a later release
    raise NotImplementedError("calc_peaked_profiles is deprecated.")
