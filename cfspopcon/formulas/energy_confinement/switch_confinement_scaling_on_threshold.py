"""Switch the confinement scaling used if below a threshold power or density."""

import xarray as xr

from ...algorithm_class import Algorithm
from ...unit_handling import Unitfull
from .solve_for_input_power import solve_energy_confinement_scaling_for_input_power


@Algorithm.register_algorithm(return_keys=["energy_confinement_time", "P_in", "SOC_LOC_ratio"])
def switch_to_linearised_ohmic_confinement_below_threshold(
    plasma_stored_energy: Unitfull,
    energy_confinement_time: Unitfull,
    P_in: Unitfull,
    average_electron_density: Unitfull,
    confinement_time_scalar: Unitfull,
    plasma_current: Unitfull,
    magnetic_field_on_axis: Unitfull,
    major_radius: Unitfull,
    areal_elongation: Unitfull,
    separatrix_elongation: Unitfull,
    inverse_aspect_ratio: Unitfull,
    average_ion_mass: Unitfull,
    triangularity_psi95: Unitfull,
    separatrix_triangularity: Unitfull,
    q_star: Unitfull,
) -> tuple[Unitfull, ...]:
    """Switch to the LOC scaling if it predicts a worse energy confinement than our selected tau_e scaling.

    Args:
        plasma_stored_energy: :term:`glossary link<plasma_stored_energy>`
        energy_confinement_time: :term:`glossary link<energy_confinement_time>`
        P_in: :term:`glossary link<P_in>`
        average_electron_density: :term:`glossary link<average_electron_density>`
        confinement_time_scalar: :term:`glossary link<confinement_time_scalar>`
        plasma_current: :term:`glossary link<plasma_current>`
        magnetic_field_on_axis: :term:`glossary link<magnetic_field_on_axis>`
        major_radius: :term:`glossary link<major_radius>`
        areal_elongation: :term:`glossary link<areal_elongation>`
        separatrix_elongation: :term:`glossary link<separatrix_elongation>`
        inverse_aspect_ratio: :term:`glossary link<inverse_aspect_ratio>`
        average_ion_mass: :term:`glossary link<average_ion_mass>`
        triangularity_psi95: :term:`glossary link<triangularity_psi95>`
        separatrix_triangularity: :term:`glossary link<separatrix_triangularity>`
        q_star: :term:`glossary link<q_star>`

    Returns:
        :term:`energy_confinement_time`, :term:`P_in`, :term:`SOC_LOC_ratio`
    """
    tau_e_LOC, P_in_LOC = solve_energy_confinement_scaling_for_input_power(
        confinement_time_scalar=confinement_time_scalar,
        plasma_current=plasma_current,
        magnetic_field_on_axis=magnetic_field_on_axis,
        average_electron_density=average_electron_density,
        major_radius=major_radius,
        areal_elongation=areal_elongation,
        separatrix_elongation=separatrix_elongation,
        inverse_aspect_ratio=inverse_aspect_ratio,
        average_ion_mass=average_ion_mass,
        triangularity_psi95=triangularity_psi95,
        separatrix_triangularity=separatrix_triangularity,
        plasma_stored_energy=plasma_stored_energy,
        q_star=q_star,
        energy_confinement_scaling="LOC",
    )

    # Use Linearized Ohmic Confinement if it gives worse energy confinement.
    SOC_LOC_ratio = energy_confinement_time / tau_e_LOC
    energy_confinement_time = xr.where(SOC_LOC_ratio > 1.0, tau_e_LOC, energy_confinement_time)  # type:ignore[no-untyped-call]
    P_in = xr.where(SOC_LOC_ratio > 1.0, P_in_LOC, P_in)  # type:ignore[no-untyped-call]

    return (energy_confinement_time, P_in, SOC_LOC_ratio)


# TODO implement switch to L-mode below PLH
