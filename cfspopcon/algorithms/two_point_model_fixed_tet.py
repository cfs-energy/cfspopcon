"""Run the two point model with a fixed sheath entrance temperature."""
from typing import Union

import xarray as xr

from ..algorithm_class import Algorithm
from ..formulas.scrape_off_layer_model import solve_target_first_two_point_model
from ..named_options import MomentumLossFunction
from ..unit_handling import Unitfull, convert_to_default_units

RETURN_KEYS = [
    "upstream_electron_temp",
    "target_electron_density",
    "SOL_power_loss_fraction",
    "target_electron_flux",
    "target_q_parallel",
]


def run_two_point_model_fixed_tet(
    target_electron_temp: Unitfull,
    q_parallel: Unitfull,
    parallel_connection_length: Unitfull,
    upstream_electron_density: Unitfull,
    toroidal_flux_expansion: Unitfull,
    fuel_average_mass_number: Unitfull,
    kappa_e0: Unitfull,
    SOL_momentum_loss_function: Union[MomentumLossFunction, xr.DataArray],
) -> dict[str, Unitfull]:
    """Run the two point model with a fixed sheath entrance temperature.

    Args:
        target_electron_temp: :term:`glossary link<target_q_parallel>`
        q_parallel: :term:`glossary link<q_parallel>`
        parallel_connection_length: :term:`glossary link<parallel_connection_length>`
        average_electron_density: :term:`glossary link<average_electron_density>`
        upstream_electron_density: :term:`glossary link<upstream_electron_density>`
        toroidal_flux_expansion: :term:`glossary link<toroidal_flux_expansion>`
        fuel_average_mass_number: :term:`glossary link<fuel_average_mass_number>`
        kappa_e0: :term:`glossary link<kappa_e0>`
        SOL_momentum_loss_function: :term:`glossary link<SOL_momentum_loss_function>`

    Returns:
        :term:`upstream_electron_temp`, :term:`target_electron_density`, :term:`SOL_power_loss_fraction`, :term:`target_electron_flux`, :term:`target_q_parallel`,
    """
    (SOL_power_loss_fraction, upstream_electron_temp, target_electron_density, target_electron_flux,) = solve_target_first_two_point_model(
        target_electron_temp=target_electron_temp,
        parallel_heat_flux_density=q_parallel,
        parallel_connection_length=parallel_connection_length,
        upstream_electron_density=upstream_electron_density,
        toroidal_flux_expansion=toroidal_flux_expansion,
        fuel_average_mass_number=fuel_average_mass_number,
        kappa_e0=kappa_e0,
        SOL_momentum_loss_function=SOL_momentum_loss_function,
    )

    target_q_parallel = q_parallel * (1.0 - SOL_power_loss_fraction)

    local_vars = locals()
    return {key: convert_to_default_units(local_vars[key], key) for key in RETURN_KEYS}


two_point_model_fixed_tet = Algorithm(
    function=run_two_point_model_fixed_tet,
    return_keys=RETURN_KEYS,
)
