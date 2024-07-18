"""Algorithms for different formulations of the extended two point model."""

from typing import Union

import xarray as xr

from ....algorithm_class import Algorithm
from ....named_options import MomentumLossFunction
from ....unit_handling import Unitfull
from .model import solve_two_point_model
from .target_first_model import solve_target_first_two_point_model


@Algorithm.register_algorithm(
    return_keys=[
        "separatrix_electron_temp",
        "target_electron_density",
        "target_electron_temp",
        "target_electron_flux",
        "target_q_parallel",
    ]
)
def two_point_model_fixed_fpow(
    SOL_power_loss_fraction: Unitfull,
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
        raise_error_if_not_converged: Raise an error if solve does not converge

    Returns:
        :term:`separatrix_electron_temp`, :term:`target_electron_density`, :term:`target_electron_temp`, :term:`target_electron_flux`, :term:`target_q_parallel`,
    """
    (
        separatrix_electron_temp,
        target_electron_density,
        target_electron_temp,
        target_electron_flux,
    ) = solve_two_point_model(
        SOL_power_loss_fraction=SOL_power_loss_fraction,
        parallel_heat_flux_density=q_parallel,
        parallel_connection_length=parallel_connection_length,
        separatrix_electron_density=nesep_over_nebar * average_electron_density,
        toroidal_flux_expansion=toroidal_flux_expansion,
        fuel_average_mass_number=fuel_average_mass_number,
        kappa_e0=kappa_e0,
        SOL_momentum_loss_function=SOL_momentum_loss_function,
        raise_error_if_not_converged=raise_error_if_not_converged,
    )

    target_q_parallel = q_parallel * (1.0 - SOL_power_loss_fraction)

    return (separatrix_electron_temp, target_electron_density, target_electron_temp, target_electron_flux, target_q_parallel)


@Algorithm.register_algorithm(
    return_keys=[
        "separatrix_electron_temp",
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
        :term:`separatrix_electron_temp`, :term:`target_electron_density`, :term:`target_electron_temp`, :term:`target_electron_flux`, :term:`SOL_power_loss_fraction`,

    """
    SOL_power_loss_fraction = (1.0 - target_q_parallel / q_parallel).clip(min=0.0, max=1.0)

    (
        separatrix_electron_temp,
        target_electron_density,
        target_electron_temp,
        target_electron_flux,
    ) = solve_two_point_model(
        SOL_power_loss_fraction=SOL_power_loss_fraction,
        parallel_heat_flux_density=q_parallel,
        parallel_connection_length=parallel_connection_length,
        separatrix_electron_density=nesep_over_nebar * average_electron_density,
        toroidal_flux_expansion=toroidal_flux_expansion,
        fuel_average_mass_number=fuel_average_mass_number,
        kappa_e0=kappa_e0,
        SOL_momentum_loss_function=SOL_momentum_loss_function,
        raise_error_if_not_converged=raise_error_if_not_converged,
    )

    return (separatrix_electron_temp, target_electron_density, target_electron_temp, target_electron_flux, SOL_power_loss_fraction)


@Algorithm.register_algorithm(
    return_keys=[
        "separatrix_electron_temp",
        "target_electron_density",
        "SOL_power_loss_fraction",
        "target_electron_flux",
        "target_q_parallel",
    ]
)
def two_point_model_fixed_tet(
    target_electron_temp: Unitfull,
    q_parallel: Unitfull,
    parallel_connection_length: Unitfull,
    separatrix_electron_density: Unitfull,
    toroidal_flux_expansion: Unitfull,
    fuel_average_mass_number: Unitfull,
    kappa_e0: Unitfull,
    SOL_momentum_loss_function: Union[MomentumLossFunction, xr.DataArray],
) -> tuple[Unitfull, ...]:
    """Run the two point model with a fixed sheath entrance temperature.

    Args:
        target_electron_temp: :term:`glossary link<target_q_parallel>`
        q_parallel: :term:`glossary link<q_parallel>`
        parallel_connection_length: :term:`glossary link<parallel_connection_length>`
        average_electron_density: :term:`glossary link<average_electron_density>`
        separatrix_electron_density: :term:`glossary link<separatrix_electron_density>`
        toroidal_flux_expansion: :term:`glossary link<toroidal_flux_expansion>`
        fuel_average_mass_number: :term:`glossary link<fuel_average_mass_number>`
        kappa_e0: :term:`glossary link<kappa_e0>`
        SOL_momentum_loss_function: :term:`glossary link<SOL_momentum_loss_function>`

    Returns:
        :term:`separatrix_electron_temp`, :term:`target_electron_density`, :term:`SOL_power_loss_fraction`, :term:`target_electron_flux`, :term:`target_q_parallel`,
    """
    (
        SOL_power_loss_fraction,
        separatrix_electron_temp,
        target_electron_density,
        target_electron_flux,
    ) = solve_target_first_two_point_model(
        target_electron_temp=target_electron_temp,
        parallel_heat_flux_density=q_parallel,
        parallel_connection_length=parallel_connection_length,
        separatrix_electron_density=separatrix_electron_density,
        toroidal_flux_expansion=toroidal_flux_expansion,
        fuel_average_mass_number=fuel_average_mass_number,
        kappa_e0=kappa_e0,
        SOL_momentum_loss_function=SOL_momentum_loss_function,
    )

    target_q_parallel = q_parallel * (1.0 - SOL_power_loss_fraction)

    return (separatrix_electron_temp, target_electron_density, SOL_power_loss_fraction, target_electron_flux, target_q_parallel)
