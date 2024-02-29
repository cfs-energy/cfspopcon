"""Calculate toroidal, poloidal, total and normalized beta."""
from .. import formulas
from ..unit_handling import Unitfull, convert_to_default_units
from .algorithm_class import Algorithm

RETURN_KEYS = [
    "beta_toroidal",
    "beta_poloidal",
    "beta",
    "normalized_beta",
]


def run_calc_beta(
    average_electron_density: Unitfull,
    average_electron_temp: Unitfull,
    average_ion_density: Unitfull,
    average_ion_temp: Unitfull,
    magnetic_field_on_axis: Unitfull,
    plasma_current: Unitfull,
    minor_radius: Unitfull,
) -> dict[str, Unitfull]:
    """Calculate toroidal, poloidal, total and normalized beta.

    Args:
        average_electron_density: :term:`glossary link<average_electron_density>`
        average_electron_temp: :term:`glossary link<average_electron_temp>`
        average_ion_density: [1e-19 m^-3] :term:`glossary link<average_ion_density>`
        average_ion_temp: :term:`glossary link<average_ion_temp>`
        magnetic_field_on_axis: :term:`glossary link<magnetic_field_on_axis>`
        plasma_current: :term:`glossary link<plasma_current>`
        minor_radius: :term:`glossary link<minor_radius>`

    Returns:
        :term:`beta_toroidal`, :term:`beta_poloidal`, :term:`beta_total`, :term:`beta_N`
    """
    beta_toroidal = formulas.calc_beta_toroidal(average_electron_density, average_electron_temp, average_ion_density, average_ion_temp, magnetic_field_on_axis)
    beta_poloidal = formulas.calc_beta_poloidal(
        average_electron_density, average_electron_temp, average_ion_density, average_ion_temp, plasma_current, minor_radius
    )

    beta = formulas.calc_beta_total(beta_toroidal=beta_toroidal, beta_poloidal=beta_poloidal)
    normalized_beta = formulas.calc_beta_normalised(beta, minor_radius, magnetic_field_on_axis, plasma_current)

    local_vars = locals()
    return {key: convert_to_default_units(local_vars[key], key) for key in RETURN_KEYS}


calc_beta = Algorithm(
    function=run_calc_beta,
    return_keys=RETURN_KEYS,
)
