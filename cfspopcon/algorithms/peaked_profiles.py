"""Calculate density peaking and the corresponding density and temperature profiles."""
from .. import formulas
from ..named_options import ProfileForm
from ..unit_handling import Unitfull, convert_to_default_units
from .algorithm_class import Algorithm

RETURN_KEYS = [
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


def run_calc_peaked_profiles(
    profile_form: ProfileForm,
    line_averaged_electron_density: Unitfull,
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
) -> dict[str, Unitfull]:
    """Calculate density peaking and the corresponding density and temperature profiles.

    Args:
        profile_form: :term:`glossary link<profile_form>`
        line_averaged_electron_density: :term:`glossary link<line_averaged_electron_density>`
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

    Returns:
    `effective_collisionality`, :term:`ion_density_peaking`, :term:`electron_density_peaking`, :term:`peak_electron_density`, :term:`peak_electron_temp`, :term:`peak_ion_temp`, :term:`rho`, :term:`electron_density_profile`, :term:`fuel_ion_density_profile`, :term:`electron_temp_profile`, :term:`ion_temp_profile`

    """
    effective_collisionality = formulas.calc_effective_collisionality(
        line_averaged_electron_density, average_electron_temp, major_radius, z_effective
    )
    ion_density_peaking = formulas.calc_density_peaking(effective_collisionality, beta_toroidal, nu_noffset=ion_density_peaking_offset)
    electron_density_peaking = formulas.calc_density_peaking(
        effective_collisionality, beta_toroidal, nu_noffset=electron_density_peaking_offset
    )

    peak_electron_density = average_electron_density * electron_density_peaking
    peak_fuel_ion_density = (
        average_electron_density * dilution * ion_density_peaking
    )  # dilution was calculated from average_electron_density in zeff_and_dilution_from_impurities.py...should it be used here (i.e. with linear densities)?
    peak_electron_temp = average_electron_temp * temperature_peaking
    peak_ion_temp = average_ion_temp * temperature_peaking

    # Calculate the total fusion power by estimating density and temperature profiles and
    # using this to calculate fusion power profiles.
    (rho, electron_density_profile, fuel_ion_density_profile, electron_temp_profile, ion_temp_profile,) = formulas.calc_1D_plasma_profiles(
        profile_form,
        line_averaged_electron_density,
        average_electron_temp,
        average_ion_temp,
        electron_density_peaking,
        ion_density_peaking,
        temperature_peaking,
        dilution,
        normalized_inverse_temp_scale_length,
    )

    local_vars = locals()
    return {key: convert_to_default_units(local_vars[key], key) for key in RETURN_KEYS}


calc_peaked_profiles = Algorithm(
    function=run_calc_peaked_profiles,
    return_keys=RETURN_KEYS,
)
