"""Calculate the impact of core impurities on z_effective and dilution."""
import xarray as xr

from .. import formulas
from ..unit_handling import Unitfull, convert_to_default_units
from .algorithm_class import Algorithm

RETURN_KEYS = [
    "impurity_charge_state",
    "z_effective",
    "dilution",
    "summed_impurity_average_density",
    "average_ion_density",
]


def run_calc_zeff_and_dilution_from_impurities(
    average_electron_density: Unitfull,
    average_electron_temp: Unitfull,
    impurities: xr.DataArray,
    atomic_data: xr.DataArray,
) -> dict[str, Unitfull]:
    """Calculate the impact of core impurities on z_effective and dilution.

    Args:
        average_electron_density: :term:`glossary link<average_electron_density>`
        average_electron_temp: :term:`glossary link<average_electron_temp>`
        impurities: :term:`glossary link<impurities>`
        atomic_data: :term:`glossary link<atomic_data>`

    Returns:
        :term:`impurity_charge_state`, :term:`z_effective`, :term:`dilution`, :term:`summed_impurity_average_density`, :term:`average_ion_density`

    """
    starting_zeff = 1.0
    starting_dilution = 1.0

    impurity_charge_state = formulas.calc_impurity_charge_state(
        average_electron_density, average_electron_temp, impurities.dim_species, atomic_data.item()
    )
    change_in_zeff = formulas.calc_change_in_zeff(impurity_charge_state, impurities)
    change_in_dilution = formulas.calc_change_in_dilution(impurity_charge_state, impurities)

    z_effective = starting_zeff + change_in_zeff.sum(dim="dim_species")
    dilution = starting_dilution - change_in_dilution.sum(dim="dim_species")
    summed_impurity_average_density = impurities.sum(dim="dim_species") * average_electron_density
    average_ion_density = dilution * average_electron_density

    local_vars = locals()
    return {key: convert_to_default_units(local_vars[key], key) for key in RETURN_KEYS}


calc_zeff_and_dilution_from_impurities = Algorithm(
    function=run_calc_zeff_and_dilution_from_impurities,
    return_keys=RETURN_KEYS,
)
