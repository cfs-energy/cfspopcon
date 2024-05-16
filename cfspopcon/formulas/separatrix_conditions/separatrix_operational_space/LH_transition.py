"""Calculate a function which is used to determine the LH transition."""
import numpy as np

from ....algorithm_class import Algorithm
from ....unit_handling import Unitfull
from .shared import (
    calc_curvature_drive,
    calc_electromagnetic_wavenumber,
    calc_electron_beta,
    calc_electron_pressure_decay_length_Eich2021H,
    calc_electron_to_ion_mass_ratio,
)


@Algorithm.register_algorithm(return_keys=["SepOS_LH_transition"])
def calc_SepOS_LH_transition(
    separatrix_electron_density: Unitfull,
    separatrix_electron_temp: Unitfull,
    major_radius: Unitfull,
    magnetic_field_on_axis: Unitfull,
    ion_mass: Unitfull,
    critical_alpha_MHD: Unitfull,
    alpha_t: Unitfull,
    poloidal_sound_larmor_radius: Unitfull,
) -> Unitfull:
    """Calculate a condition function which gives the LH transition at SepOS_LH_transition=1.

    If SepOS_LH_transition < 1, the operating point will be in L-mode
    If SepOS_LH_transition > 1, the operating point will be in H-mode

    Equation 8 from :cite:`Eich_2021`

    Args:
        separatrix_electron_density: :term:`glossary link<separatrix_electron_density>`
        separatrix_electron_temp: :term:`glossary link<separatrix_electron_temp>`
        major_radius: :term:`glossary link<major_radius>`
        magnetic_field_on_axis: :term:`glossary link<magnetic_field_on_axis>`
        ion_mass: :term:`glossary link<ion_mass>`
        critical_alpha_MHD: :term:`glossary link<critical_alpha_MHD>`
        alpha_t: :term:`glossary link<alpha_t>`
        poloidal_sound_larmor_radius: :term:`glossary link<poloidal_sound_larmor_radius>`

    Returns:
        :term:`SepOS_LH_transition`
    """
    beta_e = calc_electron_beta(separatrix_electron_density, separatrix_electron_temp, magnetic_field_on_axis)
    mu = calc_electron_to_ion_mass_ratio(ion_mass)

    electron_pressure_decay_length = calc_electron_pressure_decay_length_Eich2021H(
        alpha_t=alpha_t, poloidal_sound_larmor_radius=poloidal_sound_larmor_radius
    )

    k_EM = calc_electromagnetic_wavenumber(beta_e=beta_e, mu=mu)
    omega_B = calc_curvature_drive(perpendicular_decay_length=electron_pressure_decay_length, major_radius=major_radius)

    flow_shear_stabilisation = critical_alpha_MHD * k_EM / (1.0 + (alpha_t * k_EM / critical_alpha_MHD) ** 2)

    electron_turbulence_destabilisation = 0.5 * alpha_t
    kinetic_turbulence_destabilisation = k_EM**2 * alpha_t
    ion_turbulence_destabilisation = critical_alpha_MHD / (2.0 * k_EM**2) * np.sqrt(omega_B)
    total_destabilisation = electron_turbulence_destabilisation + ion_turbulence_destabilisation + kinetic_turbulence_destabilisation

    return flow_shear_stabilisation / total_destabilisation
