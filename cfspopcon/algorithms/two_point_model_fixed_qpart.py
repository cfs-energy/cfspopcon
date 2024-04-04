"""Run the two point model with a fixed parallel heat flux density reaching the target."""
from typing import Union

import xarray as xr

from ..algorithm_class import Algorithm
from ..deprecated_formulas.scrape_off_layer_model import solve_two_point_model
from ..named_options import MomentumLossFunction
from ..unit_handling import Unitfull


@Algorithm.register_algorithm(
    return_keys=[
        "upstream_electron_temp",
        "target_electron_density",
        "target_electron_temp",
        "target_electron_flux",
        "SOL_power_loss_fraction",
    ]
)
def two_point_model_fixed_qpart(
    target_q_parallel: Unitfull,
    q_parallel: Unitfull,
    parallel_connection_length: Unitfull,
    average_electron_density: Unitfull,
    nesep_over_nebar: Unitfull,
    toroidal_flux_expansion: Unitfull,
    fuel_average_mass_number: Unitfull,
    kappa_e0: Unitfull,
    SOL_momentum_loss_function: Union[MomentumLossFunction, xr.DataArray],
    raise_error_if_not_converged: bool = False,
) -> tuple[Unitfull, ...]:
    """Run the two point model with a fixed parallel heat flux density reaching the target.

    Args:
        target_q_parallel: :term:`glossary link<target_q_parallel>`
        q_parallel: :term:`glossary link<q_parallel>`
        parallel_connection_length: :term:`glossary link<parallel_connection_length>`
        average_electron_density: :term:`glossary link<average_electron_density>`
        nesep_over_nebar: :term:`glossary link<nesep_over_nebar>`
        toroidal_flux_expansion: :term:`glossary link<toroidal_flux_expansion>`
        fuel_average_mass_number: :term:`glossary link<fuel_average_mass_number>`
        kappa_e0: :term:`glossary link<kappa_e0>`
        SOL_momentum_loss_function: :term:`glossary link<SOL_momentum_loss_function>`
        raise_error_if_not_converged: Raise an error if solve does not converge

    Returns:
        :term:`upstream_electron_temp`, :term:`target_electron_density`, :term:`target_electron_temp`, :term:`target_electron_flux`, :term:`SOL_power_loss_fraction`,

    """
    SOL_power_loss_fraction = (1.0 - target_q_parallel / q_parallel).clip(min=0.0, max=1.0)

    (upstream_electron_temp, target_electron_density, target_electron_temp, target_electron_flux,) = solve_two_point_model(
        SOL_power_loss_fraction=SOL_power_loss_fraction,
        parallel_heat_flux_density=q_parallel,
        parallel_connection_length=parallel_connection_length,
        upstream_electron_density=nesep_over_nebar * average_electron_density,
        toroidal_flux_expansion=toroidal_flux_expansion,
        fuel_average_mass_number=fuel_average_mass_number,
        kappa_e0=kappa_e0,
        SOL_momentum_loss_function=SOL_momentum_loss_function,
        raise_error_if_not_converged=raise_error_if_not_converged,
    )

    return (upstream_electron_temp, target_electron_density, target_electron_temp, target_electron_flux, SOL_power_loss_fraction)
