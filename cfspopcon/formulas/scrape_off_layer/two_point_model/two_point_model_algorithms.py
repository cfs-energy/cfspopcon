"""Algorithms for different formulations of the extended two point model."""

from typing import Union

import xarray as xr

from ....algorithm_class import Algorithm
from ....named_options import MomentumLossFunction, ParallelConductionModel
from ....unit_handling import Unitfull, ureg
from .model import solve_two_point_model
from .target_first_model import solve_target_first_two_point_model


@Algorithm.register_algorithm(
    return_keys=[
        "separatrix_electron_temp",
        "target_electron_density",
        "target_electron_temp",
        "target_electron_flux",
        "target_q_parallel",
        "kappa_e0",
        "sheath_heat_transmission_factor",
        "Spitzer_conduction_reduction_factor",
        "upstream_SOL_collisionality",
        "delta_electron_sheath_factor",
    ]
)
def two_point_model_fixed_fpow(
    SOL_power_loss_fraction: Unitfull,
    q_parallel: Unitfull,
    parallel_connection_length: Unitfull,
    average_electron_density: Unitfull,
    nesep_over_nebar: Unitfull,
    toroidal_flux_expansion: Unitfull,
    average_ion_mass: Unitfull,
    kappa_e0: Unitfull,
    SOL_momentum_loss_function: Union[MomentumLossFunction, xr.DataArray],
    parallel_conduction_model: Union[ParallelConductionModel, xr.DataArray],
    sheath_heat_transmission_factor: Unitfull,
    flux_limit_factor_alpha: Unitfull = 0.15 * ureg.dimensionless,
    two_point_model_error_nonconverged_error: bool = False,
) -> tuple[Unitfull, ...]:
    """Run the two point model with a fixed power loss fraction in the SOL.

    Args:
        SOL_power_loss_fraction: :term:`glossary link<SOL_power_loss_fraction>`
        q_parallel: :term:`glossary link<q_parallel>`
        parallel_connection_length: :term:`glossary link<parallel_connection_length>`
        average_electron_density: :term:`glossary link<average_electron_density>`
        nesep_over_nebar: :term:`glossary link<nesep_over_nebar>`
        toroidal_flux_expansion: :term:`glossary link<toroidal_flux_expansion>`
        average_ion_mass: :term:`glossary link<average_ion_mass>`
        kappa_e0: :term:`glossary link<kappa_e0>`
        SOL_momentum_loss_function: :term:`glossary link<SOL_momentum_loss_function>`
        parallel_conduction_model :term:`glossary link<parallel_conduction_model>`
        flux_limit_factor_alpha: term:`glossary link<flux_limit_factor_alpha>`
        sheath_heat_transmission_factor: :term:`glossary link<sheath_heat_transmission_factor>`
        two_point_model_error_nonconverged_error: Raise an error if solve does not converge

    Returns:
        :term:`separatrix_electron_temp`, :term:`target_electron_density`, :term:`target_electron_temp`, :term:`target_electron_flux`, :term:`target_q_parallel`, :term:`kappa_e0', :term:'sheath_heat_transmission_factor', :term:`Spitzer_conduction_reduction_factor', :term:`upstream_SOL_collisionality', :term:`delta_electron_sheath_factor',
    """
    (
        separatrix_electron_temp,
        target_electron_density,
        target_electron_temp,
        target_electron_flux,
        kappa_e0,
        sheath_heat_transmission_factor,
        Spitzer_conduction_reduction_factor,
        upstream_SOL_collisionality,
        delta_electron_sheath_factor,
    ) = solve_two_point_model(
        SOL_power_loss_fraction=SOL_power_loss_fraction,
        q_parallel=q_parallel,
        parallel_connection_length=parallel_connection_length,
        separatrix_electron_density=nesep_over_nebar * average_electron_density,
        toroidal_flux_expansion=toroidal_flux_expansion,
        average_ion_mass=average_ion_mass,
        kappa_e0=kappa_e0,
        SOL_momentum_loss_function=SOL_momentum_loss_function,
        parallel_conduction_model=parallel_conduction_model,
        flux_limit_factor_alpha=flux_limit_factor_alpha,
        two_point_model_error_nonconverged_error=two_point_model_error_nonconverged_error,
        sheath_heat_transmission_factor=sheath_heat_transmission_factor,
    )

    target_q_parallel = q_parallel * (1.0 - SOL_power_loss_fraction)

    return (
        separatrix_electron_temp,
        target_electron_density,
        target_electron_temp,
        target_electron_flux,
        target_q_parallel,
        kappa_e0,
        sheath_heat_transmission_factor,
        Spitzer_conduction_reduction_factor,
        upstream_SOL_collisionality,
        delta_electron_sheath_factor,
    )


@Algorithm.register_algorithm(
    return_keys=[
        "separatrix_electron_temp",
        "target_electron_density",
        "target_electron_temp",
        "target_electron_flux",
        "SOL_power_loss_fraction",
        "kappa_e0",
        "sheath_heat_transmission_factor",
        "Spitzer_conduction_reduction_factor",
        "upstream_SOL_collisionality",
        "delta_electron_sheath_factor",
    ]
)
def two_point_model_fixed_qpart(
    target_q_parallel: Unitfull,
    q_parallel: Unitfull,
    parallel_connection_length: Unitfull,
    average_electron_density: Unitfull,
    nesep_over_nebar: Unitfull,
    toroidal_flux_expansion: Unitfull,
    average_ion_mass: Unitfull,
    kappa_e0: Unitfull,
    SOL_momentum_loss_function: Union[MomentumLossFunction, xr.DataArray],
    parallel_conduction_model: Union[ParallelConductionModel, xr.DataArray],
    sheath_heat_transmission_factor: Unitfull,
    flux_limit_factor_alpha: Unitfull = 0.15 * ureg.dimensionless,
    two_point_model_error_nonconverged_error: bool = False,
) -> tuple[Unitfull, ...]:
    """Run the two point model with a fixed parallel heat flux density reaching the target.

    Args:
        target_q_parallel: :term:`glossary link<target_q_parallel>`
        q_parallel: :term:`glossary link<q_parallel>`
        parallel_connection_length: :term:`glossary link<parallel_connection_length>`
        average_electron_density: :term:`glossary link<average_electron_density>`
        nesep_over_nebar: :term:`glossary link<nesep_over_nebar>`
        toroidal_flux_expansion: :term:`glossary link<toroidal_flux_expansion>`
        average_ion_mass: :term:`glossary link<average_ion_mass>`
        kappa_e0: :term:`glossary link<kappa_e0>`
        SOL_momentum_loss_function: :term:`glossary link<SOL_momentum_loss_function>`
        parallel_conduction_model :term:`glossary link<parallel_conduction_model>`
        flux_limit_factor_alpha: term:`glossary link<flux_limit_factor_alpha>`
        sheath_heat_transmission_factor: :term:`glossary link<sheath_heat_transmission_factor>`
        two_point_model_error_nonconverged_error: Raise an error if solve does not converge

    Returns:
        :term:`separatrix_electron_temp`, :term:`target_electron_density`, :term:`target_electron_temp`, :term:`target_electron_flux`, :term:`SOL_power_loss_fraction`, :term:`kappa_e0', :term:'sheath_heat_transmission_factor', :term:`Spitzer_conduction_reduction_factor', :term:`upstream_SOL_collisionality', :term:`delta_electron_sheath_factor',

    """
    SOL_power_loss_fraction = (1.0 - target_q_parallel / q_parallel).clip(min=0.0, max=1.0)

    (
        separatrix_electron_temp,
        target_electron_density,
        target_electron_temp,
        target_electron_flux,
        kappa_e0,
        sheath_heat_transmission_factor,
        Spitzer_conduction_reduction_factor,
        upstream_SOL_collisionality,
        delta_electron_sheath_factor,
    ) = solve_two_point_model(
        SOL_power_loss_fraction=SOL_power_loss_fraction,
        q_parallel=q_parallel,
        parallel_connection_length=parallel_connection_length,
        separatrix_electron_density=nesep_over_nebar * average_electron_density,
        toroidal_flux_expansion=toroidal_flux_expansion,
        average_ion_mass=average_ion_mass,
        kappa_e0=kappa_e0,
        SOL_momentum_loss_function=SOL_momentum_loss_function,
        parallel_conduction_model=parallel_conduction_model,
        flux_limit_factor_alpha=flux_limit_factor_alpha,
        two_point_model_error_nonconverged_error=two_point_model_error_nonconverged_error,
        sheath_heat_transmission_factor=sheath_heat_transmission_factor,
    )

    return (
        separatrix_electron_temp,
        target_electron_density,
        target_electron_temp,
        target_electron_flux,
        SOL_power_loss_fraction,
        kappa_e0,
        sheath_heat_transmission_factor,
        Spitzer_conduction_reduction_factor,
        upstream_SOL_collisionality,
        delta_electron_sheath_factor,
    )


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
    average_ion_mass: Unitfull,
    kappa_e0: Unitfull,
    SOL_momentum_loss_function: Union[MomentumLossFunction, xr.DataArray],
    sheath_heat_transmission_factor: Unitfull,
) -> tuple[Unitfull, ...]:
    """Run the two point model with a fixed sheath entrance temperature.

    Args:
        target_electron_temp: :term:`glossary link<target_q_parallel>`
        q_parallel: :term:`glossary link<q_parallel>`
        parallel_connection_length: :term:`glossary link<parallel_connection_length>`
        average_electron_density: :term:`glossary link<average_electron_density>`
        separatrix_electron_density: :term:`glossary link<separatrix_electron_density>`
        toroidal_flux_expansion: :term:`glossary link<toroidal_flux_expansion>`
        average_ion_mass: :term:`glossary link<average_ion_mass>`
        kappa_e0: :term:`glossary link<kappa_e0>`
        SOL_momentum_loss_function: :term:`glossary link<SOL_momentum_loss_function>`
        sheath_heat_transmission_factor: :term:`glossary link<sheath_heat_transmission_factor>`

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
        q_parallel=q_parallel,
        parallel_connection_length=parallel_connection_length,
        separatrix_electron_density=separatrix_electron_density,
        toroidal_flux_expansion=toroidal_flux_expansion,
        average_ion_mass=average_ion_mass,
        kappa_e0=kappa_e0,
        SOL_momentum_loss_function=SOL_momentum_loss_function,
        sheath_heat_transmission_factor=sheath_heat_transmission_factor,
    )

    target_q_parallel = q_parallel * (1.0 - SOL_power_loss_fraction)

    return (separatrix_electron_temp, target_electron_density, SOL_power_loss_fraction, target_electron_flux, target_q_parallel)
