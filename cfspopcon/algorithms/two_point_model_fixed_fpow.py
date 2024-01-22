"""Run the two point model with a fixed power loss fraction in the SOL."""
from typing import Union

import xarray as xr

from ..formulas.scrape_off_layer_model import solve_two_point_model
from ..named_options import MomentumLossFunction, ParallelConductionModel
from ..unit_handling import Quantity, Unitfull, convert_to_default_units, ureg
from .algorithm_class import Algorithm

RETURN_KEYS = [
    "upstream_electron_temp",
    "target_electron_density",
    "target_electron_temp",
    "target_electron_flux",
    "Spitzer_conduction_reduction_factor",
    "upstream_SOL_collisionality",
    "target_q_parallel",
]


def run_two_point_model_fixed_fpow(
    SOL_power_loss_fraction: Union[float, xr.DataArray],
    q_parallel: Union[Quantity, xr.DataArray],
    parallel_connection_length: Union[Quantity, xr.DataArray],
    average_electron_density: Union[Quantity, xr.DataArray],
    nesep_over_nebar: Union[float, xr.DataArray],
    toroidal_flux_expansion: Union[float, xr.DataArray],
    fuel_average_mass_number: Union[Quantity, xr.DataArray],
    kappa_e0: Union[Quantity, xr.DataArray],
    SOL_momentum_loss_function: Union[MomentumLossFunction, xr.DataArray],
    parallel_conduction_model: Union[ParallelConductionModel, xr.DataArray],
    flux_limit_factor_alpha: Unitfull = 0.15 * ureg.dimensionless,
    raise_error_if_not_converged: bool = False,
) -> dict[str, Union[float, Quantity, xr.DataArray]]:
    """Run the two point model with a fixed power loss fraction in the SOL.

    Args:
        SOL_power_loss_fraction: :term:`glossary link<SOL_power_loss_fraction>`
        q_parallel: :term:`glossary link<q_parallel>`
        parallel_connection_length: :term:`glossary link<parallel_connection_length>`
        average_electron_density: :term:`glossary link<average_electron_density>`
        nesep_over_nebar: :term:`glossary link<nesep_over_nebar>`
        toroidal_flux_expansion: :term:`glossary link<toroidal_flux_expansion>`
        fuel_average_mass_number: :term:`glossary link<fuel_average_mass_number>`
        kappa_e0: :term:`glossary link<kappa_e0>`
        SOL_momentum_loss_function: :term:`glossary link<SOL_momentum_loss_function>`
        parallel_conduction_model :term:`glossary link<parallel_conduction_model>`
        flux_limit_factor_alpha: term:`glossary link<flux_limit_factor_alpha>`
        raise_error_if_not_converged: Raise an error if solve does not converge

    Returns:
        :term:`upstream_electron_temp`, :term:`target_electron_density`, :term:`target_electron_temp`, :term:`target_electron_flux`, :term:`target_q_parallel`,
    """
    (
        upstream_electron_temp,
        target_electron_density,
        target_electron_temp,
        target_electron_flux,
        Spitzer_conduction_reduction_factor,
        upstream_SOL_collisionality,
    ) = solve_two_point_model(
        SOL_power_loss_fraction=SOL_power_loss_fraction,
        parallel_heat_flux_density=q_parallel,
        parallel_connection_length=parallel_connection_length,
        upstream_electron_density=nesep_over_nebar * average_electron_density,
        toroidal_flux_expansion=toroidal_flux_expansion,
        fuel_average_mass_number=fuel_average_mass_number,
        kappa_e0=kappa_e0,
        SOL_momentum_loss_function=SOL_momentum_loss_function,
        parallel_conduction_model=parallel_conduction_model,
        flux_limit_factor_alpha=flux_limit_factor_alpha,
        raise_error_if_not_converged=raise_error_if_not_converged,
    )

    target_q_parallel = q_parallel * (1.0 - SOL_power_loss_fraction)

    local_vars = locals()
    return {key: convert_to_default_units(local_vars[key], key) for key in RETURN_KEYS}


two_point_model_fixed_fpow = Algorithm(
    function=run_two_point_model_fixed_fpow,
    return_keys=RETURN_KEYS,
)
