"""Calculate toroidal, poloidal, total and normalized beta."""
from .. import deprecated_formulas
from ..algorithm_class import Algorithm
from ..unit_handling import Unitfull


@Algorithm.register_algorithm(
    return_keys=[
        "beta_toroidal",
        "beta_poloidal",
        "beta",
        "normalized_beta",
    ]
)
def calc_beta(
    average_electron_density: Unitfull,
    average_electron_temp: Unitfull,
    average_ion_temp: Unitfull,
    magnetic_field_on_axis: Unitfull,
    plasma_current: Unitfull,
    minor_radius: Unitfull,
) -> tuple[Unitfull, ...]:
    """Calculate toroidal, poloidal, total and normalized beta.

    Args:
        average_electron_density: :term:`glossary link<average_electron_density>`
        average_electron_temp: :term:`glossary link<average_electron_temp>`
        average_ion_temp: :term:`glossary link<average_ion_temp>`
        magnetic_field_on_axis: :term:`glossary link<magnetic_field_on_axis>`
        plasma_current: :term:`glossary link<plasma_current>`
        minor_radius: :term:`glossary link<minor_radius>`

    Returns:
        :term:`beta_toroidal`, :term:`beta_poloidal`, :term:`beta_total`, :term:`beta_N`
    """
    beta_toroidal = deprecated_formulas.calc_beta_toroidal(
        average_electron_density, average_electron_temp, average_ion_temp, magnetic_field_on_axis
    )
    beta_poloidal = deprecated_formulas.calc_beta_poloidal(
        average_electron_density, average_electron_temp, average_ion_temp, plasma_current, minor_radius
    )

    beta = deprecated_formulas.calc_beta_total(beta_toroidal=beta_toroidal, beta_poloidal=beta_poloidal)
    normalized_beta = deprecated_formulas.calc_beta_normalised(beta, minor_radius, magnetic_field_on_axis, plasma_current)

    return beta_toroidal, beta_poloidal, beta, normalized_beta
