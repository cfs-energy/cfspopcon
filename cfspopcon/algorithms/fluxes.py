"""Calculate PF flux contribution and resistive, internal, and external flux consumed over the ramp-up."""
from .. import formulas
from ..unit_handling import Unitfull, convert_to_default_units
from .algorithm_class import Algorithm

RETURN_KEYS = [
    "internal_flux",
    "external_flux",
    "resistive_flux",
    "poloidal_field_flux",
    "max_flux_for_flattop",
    "max_flattop_duration",
    "breakdown_flux_consumption",
    "flux_needed_from_CS_over_rampup",
]


def run_calc_fluxes(
    plasma_current: Unitfull,
    major_radius: Unitfull,
    internal_inductance: Unitfull,
    external_inductance: Unitfull,
    ejima_coefficient: Unitfull,
    vertical_field_mutual_inductance: Unitfull,
    vertical_magnetic_field: Unitfull,
    loop_voltage: Unitfull,
    total_flux_available_from_CS: Unitfull,
) -> dict[str, Unitfull]:
    """Calculate PF flux contribution and resistive, internal, and external flux consumed over the ramp-up.

    Args:
        plasma_current: :term:`glossary link<plasma_current>`
        major_radius: :term:`glossary link<major_radius>`
        internal_inductance: :term:`glossary link<internal_inductance>`
        external_inductance: :term:`glossary link<external_inductance>`
        vertical_field_mutual_inductance: :term:`glossary link<vertical_field_mutual_inductance>`
        vertical_magnetic_field: :term:`glossary link<vertical_magnetic_field>`
        ejima_coefficient: :term:`glossary link<ejima_coefficient>`
        loop_voltage: :term:`glossary link<loop_voltage>`
        total_flux_available_from_CS: :term:`glossary link<total_flux_available_from_CS>`

    Returns:
        :term:`resistive_flux`, :term:`internal_flux`, :term:`external_flux`, :term:`max_flattop_duration`, :term:`max_flux_for_flattop`, :term:`breakdown_flux_consumption`, :term:`glossary link<flux_needed_from_CS_over_rampup>`
    """
    internal_flux = formulas.calc_flux_internal(plasma_current, internal_inductance)
    external_flux = formulas.calc_flux_external(plasma_current, external_inductance)
    resistive_flux = formulas.calc_flux_res(plasma_current, major_radius, ejima_coefficient)
    poloidal_field_flux = formulas.calc_flux_PF(vertical_field_mutual_inductance, vertical_magnetic_field, major_radius)

    flux_needed_from_CS_over_rampup = internal_flux + external_flux + resistive_flux - poloidal_field_flux
    max_flux_for_flattop = total_flux_available_from_CS - internal_flux - external_flux - resistive_flux + poloidal_field_flux
    max_flattop_duration = max_flux_for_flattop / loop_voltage

    breakdown_flux_consumption = formulas.calc_breakdown_flux_consumption(major_radius)

    local_vars = locals()
    return {key: convert_to_default_units(local_vars[key], key) for key in RETURN_KEYS}


calc_fluxes = Algorithm(
    function=run_calc_fluxes,
    return_keys=RETURN_KEYS,
)
