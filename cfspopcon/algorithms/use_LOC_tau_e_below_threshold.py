"""Switch to the LOC scaling if it predicts a worse energy confinement than our selected tau_e scaling."""
import xarray as xr

from .. import formulas, named_options
from ..unit_handling import Unitfull, convert_to_default_units
from .algorithm_class import Algorithm

RETURN_KEYS = [
    "energy_confinement_time",
    "P_in",
    "SOC_LOC_ratio",
]


def run_use_LOC_tau_e_below_threshold(
    plasma_stored_energy: Unitfull,
    energy_confinement_time: Unitfull,
    P_in: Unitfull,
    line_averaged_electron_density: Unitfull,
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
) -> dict[str, Unitfull]:
    """Switch to the LOC scaling if it predicts a worse energy confinement than our selected tau_e scaling.

    Args:
        plasma_stored_energy: :term:`glossary link<plasma_stored_energy>`
        energy_confinement_time: :term:`glossary link<energy_confinement_time>`
        P_in: :term:`glossary link<P_in>`
        line_averaged_electron_density: :term:`glossary link<line_averaged_electron_density>`
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

    Returns:
    :term:`energy_confinement_time`, :term:`P_in`, :term:`SOC_LOC_ratio`

    """
    # Calculate linear ohmic confinement for low density
    energy_confinement_time_LOC, P_in_LOC = formulas.calc_tau_e_and_P_in_from_scaling(
        confinement_time_scalar=confinement_time_scalar,
        plasma_current=plasma_current,
        magnetic_field_on_axis=magnetic_field_on_axis,
        average_electron_density=line_averaged_electron_density,
        major_radius=major_radius,
        areal_elongation=areal_elongation,
        separatrix_elongation=separatrix_elongation,
        inverse_aspect_ratio=inverse_aspect_ratio,
        fuel_average_mass_number=fuel_average_mass_number,
        triangularity_psi95=triangularity_psi95,
        separatrix_triangularity=separatrix_triangularity,
        plasma_stored_energy=plasma_stored_energy,
        q_star=q_star,
        tau_e_scaling=named_options.ConfinementScaling.LOC,
    )

    # Use Linearized Ohmic Confinement if it gives worse energy confinement.
    SOC_LOC_ratio = energy_confinement_time / energy_confinement_time_LOC
    energy_confinement_time = xr.where(
        SOC_LOC_ratio > 1.0, energy_confinement_time_LOC, energy_confinement_time
    )  # type:ignore[no-untyped-call]
    P_in = xr.where(SOC_LOC_ratio > 1.0, P_in_LOC, P_in)  # type:ignore[no-untyped-call]

    local_vars = locals()
    return {key: convert_to_default_units(local_vars[key], key) for key in RETURN_KEYS}


use_LOC_tau_e_below_threshold = Algorithm(
    function=run_use_LOC_tau_e_below_threshold,
    return_keys=RETURN_KEYS,
)
