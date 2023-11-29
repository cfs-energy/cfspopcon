"""Calculate plasma current from edge safety factor."""
import numpy as np

from ..formulas.separatrix_operational_space import (
    calc_alpha_t_with_fixed_coulomb_log,
    calc_critical_MHD_parameter_alpha_c,
    calc_cylindrical_edge_safety_factor,
    calc_ideal_MHD_limit_condition,
    calc_ideal_MHD_limit_condition_with_alpha_MHD,
    calc_L_mode_density_limit_condition,
    calc_lambda_pe_Eich2021H,
    calc_lambda_pe_Manz2023L,
    calc_LH_transition_condition_transition_condition,
    calc_sound_larmor_radius_rho_s,
)
from ..unit_handling import Unitfull, convert_to_default_units, ureg
from .algorithm_class import Algorithm

RETURN_KEYS = [
    "Lmode_density_limit_condition",
    "ideal_MHD_limit_condition",
    "LH_transition_condition",
    "alpha_t_turbulence_param",
]


def run_calc_separatrix_operational_space(
    separatrix_electron_density: Unitfull,
    separatrix_electron_temp: Unitfull,
    magnetic_field_on_axis: Unitfull,
    major_radius: Unitfull,
    minor_radius: Unitfull,
    ion_mass: Unitfull,
    plasma_current: Unitfull,
    elongation_psi95: Unitfull,
    triangularity_psi95: Unitfull,
    z_effective: Unitfull,
    mean_ion_charge: Unitfull,
    ion_to_electron_temp_ratio: Unitfull = 1.0,
) -> dict[str, Unitfull]:
    """Calculate plasma current from edge safety factor.

    Args:
        separatrix_electron_density: :term:`glossary link<separatrix_electron_density>`
        separatrix_electron_temp: :term:`glossary link<separatrix_electron_temp>`
        magnetic_field_on_axis: :term:`glossary link<magnetic_field_on_axis>`
        major_radius: :term:`glossary link<major_radius>`
        minor_radius: :term:`glossary link<minor_radius>`
        ion_mass: :term:`glossary link<ion_mass>`
        plasma_current: :term:`glossary link<plasma_current>`
        elongation_psi95: :term:`glossary link<elongation_psi95>`
        triangularity_psi95: :term:`glossary link<triangularity_psi95>`
        z_effective: :term:`glossary link<z_effective>`
        mean_ion_charge: :term:`glossary link<mean_ion_charge>`
        ion_to_electron_temp_ratio: :term:`glossary link<ion_to_electron_temp_ratio>`

    Returns:
        :term:`Lmode_density_limit_condition`, :term:`ideal_MHD_limit_condition`, :term:`LH_transition_condition`, :term:`alpha_t_turbulence_param`
    """
    edge_safety_factor = calc_cylindrical_edge_safety_factor(
        major_radius=major_radius,
        minor_radius=minor_radius,
        elongation=elongation_psi95,
        triangularity=triangularity_psi95,
        magnetic_field_strength=magnetic_field_on_axis,
        plasma_current=plasma_current,
    )

    alpha_t_turbulence_param = calc_alpha_t_with_fixed_coulomb_log(
        electron_density=separatrix_electron_density,
        electron_temperature=separatrix_electron_temp,
        edge_safety_factor=edge_safety_factor,
        major_radius=major_radius,
        ion_mass=ion_mass,
        Zeff=z_effective,
        Z=mean_ion_charge,
        ion_to_electron_temperature_ratio=ion_to_electron_temp_ratio,
    )
    alpha_c = calc_critical_MHD_parameter_alpha_c(elongation=elongation_psi95, triangularity=triangularity_psi95)

    poloidal_circumference = 2.0 * np.pi * minor_radius * (1 + 0.55 * (elongation_psi95 - 1)) * (1 + 0.08 * triangularity_psi95**2)
    B_pol_avg = ureg.mu_0 * plasma_current / poloidal_circumference
    rho_s_pol = calc_sound_larmor_radius_rho_s(
        electron_temperature=separatrix_electron_temp, magnetic_field_strength=B_pol_avg, ion_mass=ion_mass
    )

    lambda_pe_L = calc_lambda_pe_Manz2023L(alpha_t_turbulence_param)
    lambda_pe_H = calc_lambda_pe_Eich2021H(alpha_t_turbulence_param, rho_s_pol=rho_s_pol, factor=3.6)

    Lmode_density_limit_condition = calc_L_mode_density_limit_condition(
        separatrix_electron_density,
        separatrix_electron_temp,
        major_radius,
        magnetic_field_on_axis,
        ion_mass,
        alpha_c,
        alpha_t_turbulence_param,
        lambda_pe=lambda_pe_L,
    )
    ideal_MHD_limit_condition = calc_ideal_MHD_limit_condition(
        separatrix_electron_density,
        separatrix_electron_temp,
        major_radius,
        magnetic_field_on_axis,
        edge_safety_factor,
        alpha_c,
        alpha_t_turbulence_param,
        lambda_pe=lambda_pe_H,
    )
    ideal_MHD_limit_condition2 = calc_ideal_MHD_limit_condition_with_alpha_MHD(
        separatrix_electron_density,
        separatrix_electron_temp,
        major_radius,
        magnetic_field_on_axis,
        edge_safety_factor,
        alpha_c,
        lambda_pe=lambda_pe_H,
    )
    LH_transition_condition = calc_LH_transition_condition_transition_condition(
        separatrix_electron_density,
        separatrix_electron_temp,
        major_radius,
        magnetic_field_on_axis,
        ion_mass,
        alpha_c,
        alpha_t_turbulence_param,
        lambda_pe=lambda_pe_H,
    )

    local_vars = locals()
    return {key: convert_to_default_units(local_vars[key], key) for key in RETURN_KEYS}


calc_separatrix_operational_space = Algorithm(
    function=run_calc_separatrix_operational_space,
    return_keys=RETURN_KEYS,
)
