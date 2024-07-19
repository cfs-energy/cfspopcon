"""Calculate a function which is used to determine the L-mode density limit."""

from ....algorithm_class import Algorithm
from ....unit_handling import Unitfull
from .shared import (
    calc_curvature_drive,
    calc_electromagnetic_wavenumber,
    calc_electron_beta,
    calc_electron_pressure_decay_length_Manz2023L,
    calc_electron_to_ion_mass_ratio,
    calc_resistive_ballooning_wavenumber,
)


@Algorithm.register_algorithm(return_keys=["SepOS_density_limit"])
def calc_SepOS_L_mode_density_limit(
    separatrix_electron_density: Unitfull,
    separatrix_electron_temp: Unitfull,
    major_radius: Unitfull,
    magnetic_field_on_axis: Unitfull,
    ion_mass: Unitfull,
    critical_alpha_MHD: Unitfull,
    alpha_t: Unitfull,
) -> Unitfull:
    """Calculate a condition function which gives the L-mode density limit when SepOS_density_limit=1.

    If SepOS_density_limit < 1, the operating point is stable
    If SepOS_density_limit > 1, the operating point will disrupt if not above the LH transition

    Equation 3 from :cite:`Eich_2021`

    Args:
        separatrix_electron_density: :term:`glossary link<separatrix_electron_density>`
        separatrix_electron_temp: :term:`glossary link<separatrix_electron_temp>`
        major_radius: :term:`glossary link<major_radius>`
        magnetic_field_on_axis: :term:`glossary link<magnetic_field_on_axis>`
        ion_mass: :term:`glossary link<ion_mass>`
        critical_alpha_MHD: :term:`glossary link<critical_alpha_MHD>`
        alpha_t: :term:`glossary link<alpha_t>`

    Returns:
        :term:`SepOS_density_limit`
    """
    electron_pressure_decay_length = calc_electron_pressure_decay_length_Manz2023L(alpha_t=alpha_t)

    omega_B = calc_curvature_drive(perpendicular_decay_length=electron_pressure_decay_length, major_radius=major_radius)
    beta_e = calc_electron_beta(
        electron_density=separatrix_electron_density, electron_temp=separatrix_electron_temp, magnetic_field_strength=magnetic_field_on_axis
    )
    mu = calc_electron_to_ion_mass_ratio(ion_mass=ion_mass)

    k_EM = calc_electromagnetic_wavenumber(beta_e=beta_e, mu=mu)
    k_RBM = calc_resistive_ballooning_wavenumber(critical_alpha_MHD=critical_alpha_MHD, alpha_t=alpha_t, omega_B=omega_B)

    return k_EM / k_RBM
