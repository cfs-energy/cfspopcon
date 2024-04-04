"""Calculate the input power required to maintain the stored energy, given a tau_e scaling."""
from .. import deprecated_formulas, named_options
from ..algorithm_class import Algorithm
from ..unit_handling import Unitfull


@Algorithm.register_algorithm(
    return_keys=[
        "energy_confinement_time",
        "P_in",
    ]
)
def calc_power_balance_from_tau_e(
    plasma_stored_energy: Unitfull,
    average_electron_density: Unitfull,
    confinement_time_scalar: Unitfull,
    plasma_current: Unitfull,
    magnetic_field_on_axis: Unitfull,
    major_radius: Unitfull,
    areal_elongation: Unitfull,
    separatrix_elongation: Unitfull,
    inverse_aspect_ratio: Unitfull,
    fuel_average_mass_number: Unitfull,
    triangularity_psi95: Unitfull,
    separatrix_triangularity: Unitfull,
    q_star: Unitfull,
    energy_confinement_scaling: named_options.ConfinementScaling,
) -> tuple[Unitfull, ...]:
    """Calculate the input power required to maintain the stored energy, given a tau_e scaling.

    Args:
        plasma_stored_energy: :term:`glossary link<plasma_stored_energy>`
        average_electron_density: :term:`glossary link<average_electron_density>`
        confinement_time_scalar: :term:`glossary link<confinement_time_scalar>`
        plasma_current: :term:`glossary link<plasma_current>`
        magnetic_field_on_axis: :term:`glossary link<magnetic_field_on_axis>`
        major_radius: :term:`glossary link<major_radius>`
        areal_elongation: :term:`glossary link<areal_elongation>`
        separatrix_elongation: :term:`glossary link<separatrix_elongation>`
        inverse_aspect_ratio: :term:`glossary link<inverse_aspect_ratio>`
        fuel_average_mass_number: :term:`glossary link<fuel_average_mass_number>`
        triangularity_psi95: :term:`glossary link<triangularity_psi95>`
        separatrix_triangularity: :term:`glossary link<separatrix_triangularity>`
        q_star: :term:`glossary link<q_star>`
        energy_confinement_scaling: :term:`glossary link<energy_confinement_scaling>`

    Returns:
    :term:`energy_confinement_time`, :term:`P_in`

    """
    energy_confinement_time, P_in = deprecated_formulas.calc_tau_e_and_P_in_from_scaling(
        confinement_time_scalar=confinement_time_scalar,
        plasma_current=plasma_current,
        magnetic_field_on_axis=magnetic_field_on_axis,
        average_electron_density=average_electron_density,
        major_radius=major_radius,
        areal_elongation=areal_elongation,
        separatrix_elongation=separatrix_elongation,
        inverse_aspect_ratio=inverse_aspect_ratio,
        fuel_average_mass_number=fuel_average_mass_number,
        triangularity_psi95=triangularity_psi95,
        separatrix_triangularity=separatrix_triangularity,
        plasma_stored_energy=plasma_stored_energy,
        q_star=q_star,
        tau_e_scaling=energy_confinement_scaling,
    )

    return (energy_confinement_time, P_in)
