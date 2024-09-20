"""Compute all terms in the two-point-model for a fixed SOL power loss fraction."""

from typing import Union

import numpy as np
import xarray as xr

from ....named_options import MomentumLossFunction, ParallelConductionModel
from ....unit_handling import Unitfull, ureg
from ..separatrix_electron_temp import calc_separatrix_electron_temp
from .momentum_loss_functions import calc_SOL_momentum_loss_fraction
from .parallel_conduction import (
    calc_delta_electron_sheath_factor,
    calc_Spitzer_conduction_reduction_factor_fluxlim,
    calc_Spitzer_conduction_reduction_factor_scaling,
)
from .separatrix_pressure import calc_upstream_total_pressure
from .target_electron_density import (
    calc_f_other_target_electron_density,
    calc_f_vol_loss_target_electron_density,
    calc_target_electron_density,
    calc_target_electron_density_basic,
)
from .target_electron_flux import (
    calc_f_other_target_electron_flux,
    calc_f_vol_loss_target_electron_flux,
    calc_target_electron_flux,
    calc_target_electron_flux_basic,
)
from .target_electron_temp import (
    calc_f_other_target_electron_temp,
    calc_f_vol_loss_target_electron_temp,
    calc_target_electron_temp,
    calc_target_electron_temp_basic,
)
from .upstream_SOL_collisionality import calc_upstream_SOL_collisionality


def solve_two_point_model(
    SOL_power_loss_fraction: Unitfull,
    q_parallel: Unitfull,
    parallel_connection_length: Unitfull,
    separatrix_electron_density: Unitfull,
    toroidal_flux_expansion: Unitfull,
    average_ion_mass: Unitfull,
    kappa_e0: Unitfull,
    SOL_momentum_loss_function: Union[MomentumLossFunction, xr.DataArray],
    parallel_conduction_model: Union[ParallelConductionModel, xr.DataArray],
    initial_target_electron_temp: Unitfull = 10.0 * ureg.eV,
    sheath_heat_transmission_factor: Unitfull = 7.5 * ureg.dimensionless,
    SOL_conduction_fraction: Unitfull = 1.0 * ureg.dimensionless,
    target_ratio_of_ion_to_electron_temp: Unitfull = 1.0 * ureg.dimensionless,
    target_ratio_of_electron_to_ion_density: Unitfull = 1.0 * ureg.dimensionless,
    target_mach_number: Unitfull = 1.0 * ureg.dimensionless,
    upstream_ratio_of_ion_to_electron_temp: Unitfull = 1.0 * ureg.dimensionless,
    upstream_ratio_of_electron_to_ion_density: Unitfull = 1.0 * ureg.dimensionless,
    upstream_mach_number: Unitfull = 0.0 * ureg.dimensionless,
    delta_electron_sheath_factor: Unitfull = 0.0 * ureg.dimensionless,
    Spitzer_conduction_reduction_factor: Unitfull = 1.0 * ureg.dimensionless,
    flux_limit_factor_alpha: Unitfull = 0.15 * ureg.dimensionless,
    # Controlling the iterative solve
    max_iterations: int = 100,
    upstream_temp_relaxation: float = 0.5,
    target_electron_density_relaxation: float = 0.5,
    target_temp_relaxation: float = 0.5,
    upstream_temp_max_residual: float = 1e-2,
    target_electron_density_max_residual: float = 1e-2,
    target_temp_max_residual: float = 1e-2,
    # Return converged values even if not whole array is converged
    two_point_model_error_nonconverged_error: bool = True,
    # Print information about the solve to terminal
    quiet: bool = True,
) -> tuple[
    Unitfull,
    Unitfull,
    Unitfull,
    Unitfull,
    Unitfull,
    Unitfull,
    Unitfull,
    Unitfull,
    Unitfull,
]:
    """Calculate the upstream and target electron temperature and target electron density according to the extended two-point-model.

    Args:
        SOL_power_loss_fraction: [~]
        q_parallel: [GW/m^2]
        parallel_connection_length: [m]
        separatrix_electron_density: [m^-3]
        toroidal_flux_expansion: [~]
        average_ion_mass: [~]
        kappa_e0: electron heat conductivity constant [W / (eV^3.5 * m)]
        SOL_momentum_loss_function: which momentum loss function to use
        parallel_conduction_model: which model for the parallel heat flux conduction to use
        initial_target_electron_temp: starting guess for target electron temp [eV]
        sheath_heat_transmission_factor: [~]
        SOL_conduction_fraction: [~]
        target_ratio_of_ion_to_electron_temp: [~]
        target_ratio_of_electron_to_ion_density: [~]
        target_mach_number: [~]
        upstream_ratio_of_ion_to_electron_temp: [~]
        upstream_ratio_of_electron_to_ion_density: [~]
        upstream_mach_number: [~]
        delta_electron_sheath_factor: Increase in electron sheath transmission factor from kinetic effects
        Spitzer_conduction_reduction_factor: multiplication factor on kappa_e0 to account for kinetic effects on reduced conduction
        flux_limit_factor_alpha: mutliplication factor on the "free-streaming-flux" quantity used by the flux limited conductivity
        max_iterations: how many iterations to try before returning NaN
        upstream_temp_relaxation: step-size for upstream Te evolution
        target_electron_density_relaxation: step-size for target ne evolution
        target_temp_relaxation: step-size for target Te evolution
        upstream_temp_max_residual: relative rate of change for convergence for upstream Te evolution
        target_electron_density_max_residual: relative rate of change for convergence for target ne evolution
        target_temp_max_residual: relative rate of change for convergence for target Te evolution
        two_point_model_error_nonconverged_error: raise an error if not all point converge within max iterations (otherwise return NaN)
        quiet: if not True, print additional information about the iterative solve to terminal
    Returns:
        separatrix_electron_temp [eV], target_electron_density [m^-3], target_electron_temp [eV], target_electron_flux [m^-2 s^-1]
    """
    f_other_kwargs = dict(
        target_ratio_of_ion_to_electron_temp=target_ratio_of_ion_to_electron_temp,
        target_ratio_of_electron_to_ion_density=target_ratio_of_electron_to_ion_density,
        target_mach_number=target_mach_number,
        toroidal_flux_expansion=toroidal_flux_expansion,
    )
    f_other_target_electron_density = calc_f_other_target_electron_density(**f_other_kwargs)
    f_other_target_electron_temp = calc_f_other_target_electron_temp(**f_other_kwargs)
    f_other_target_electron_flux = calc_f_other_target_electron_flux(**f_other_kwargs)

    iteration = 0
    target_electron_temp = initial_target_electron_temp
    separatrix_electron_temp = 100

    while iteration < max_iterations:
        iteration += 1

        if parallel_conduction_model == ParallelConductionModel.KineticCorrectionScalings:
            if iteration != 1:
                upstream_SOL_collisionality = calc_upstream_SOL_collisionality(
                    separatrix_electron_density=separatrix_electron_density,
                    separatrix_electron_temp=separatrix_electron_temp,
                    parallel_connection_length=parallel_connection_length,
                )

                Spitzer_conduction_reduction_factor = calc_Spitzer_conduction_reduction_factor_scaling(
                    upstream_SOL_collisionality=upstream_SOL_collisionality,
                )

                delta_electron_sheath_factor = calc_delta_electron_sheath_factor(
                    separatrix_electron_temp=separatrix_electron_temp,
                    target_electron_temp=target_electron_temp,
                    SOL_momentum_loss_fraction=calc_SOL_momentum_loss_fraction(SOL_momentum_loss_function, target_electron_temp),
                )

        if parallel_conduction_model == ParallelConductionModel.FluxLimiter:
            if iteration != 1:
                Spitzer_conduction_reduction_factor = calc_Spitzer_conduction_reduction_factor_fluxlim(
                    separatrix_electron_density=separatrix_electron_density,
                    separatrix_electron_temp=separatrix_electron_temp,
                    parallel_connection_length=parallel_connection_length,
                    target_electron_temp=target_electron_temp,
                    SOL_conduction_fraction=SOL_conduction_fraction,
                    kappa_e0=kappa_e0,
                    flux_limit_factor_alpha=flux_limit_factor_alpha,
                )

        new_separatrix_electron_temp = calc_separatrix_electron_temp(
            target_electron_temp=target_electron_temp,
            q_parallel=q_parallel,
            parallel_connection_length=parallel_connection_length,
            SOL_conduction_fraction=SOL_conduction_fraction,
            kappa_e0=Spitzer_conduction_reduction_factor * kappa_e0,
        )

        upstream_total_pressure = calc_upstream_total_pressure(
            separatrix_electron_density=separatrix_electron_density,
            separatrix_electron_temp=new_separatrix_electron_temp,
            upstream_ratio_of_ion_to_electron_temp=upstream_ratio_of_ion_to_electron_temp,
            upstream_ratio_of_electron_to_ion_density=upstream_ratio_of_electron_to_ion_density,
            upstream_mach_number=upstream_mach_number,
        )

        f_vol_loss_kwargs = dict(
            SOL_power_loss_fraction=SOL_power_loss_fraction,
            SOL_momentum_loss_fraction=calc_SOL_momentum_loss_fraction(SOL_momentum_loss_function, target_electron_temp),
        )

        f_basic_kwargs = dict(
            average_ion_mass=average_ion_mass,
            q_parallel=q_parallel,
            upstream_total_pressure=upstream_total_pressure,
            sheath_heat_transmission_factor=sheath_heat_transmission_factor + delta_electron_sheath_factor,
        )

        target_electron_density_basic = calc_target_electron_density_basic(**f_basic_kwargs)
        target_electron_temp_basic = calc_target_electron_temp_basic(**f_basic_kwargs)

        f_vol_loss_target_electron_density = calc_f_vol_loss_target_electron_density(**f_vol_loss_kwargs)
        f_vol_loss_target_electron_temp = calc_f_vol_loss_target_electron_temp(**f_vol_loss_kwargs)

        new_target_electron_density = calc_target_electron_density(
            target_electron_density_basic=target_electron_density_basic,
            f_vol_loss_target_electron_density=f_vol_loss_target_electron_density,
            f_other_target_electron_density=f_other_target_electron_density,
        )

        new_target_electron_temp = calc_target_electron_temp(
            target_electron_temp_basic=target_electron_temp_basic,
            f_vol_loss_target_electron_temp=f_vol_loss_target_electron_temp,
            f_other_target_electron_temp=f_other_target_electron_temp,
        )

        if iteration == 1:
            separatrix_electron_temp = new_separatrix_electron_temp
            target_electron_density = new_target_electron_density
            target_electron_temp = new_target_electron_temp
            continue

        change_in_separatrix_electron_temp = new_separatrix_electron_temp - separatrix_electron_temp
        change_in_target_electron_density = new_target_electron_density - target_electron_density
        change_in_target_electron_temp = new_target_electron_temp - target_electron_temp

        separatrix_electron_temp = separatrix_electron_temp + upstream_temp_relaxation * change_in_separatrix_electron_temp
        target_electron_density = target_electron_density + target_electron_density_relaxation * change_in_target_electron_density
        target_electron_temp = target_electron_temp + target_temp_relaxation * change_in_target_electron_temp

        if np.all(
            [
                np.abs(change_in_separatrix_electron_temp / separatrix_electron_temp).max() < upstream_temp_max_residual,
                np.abs(change_in_target_electron_density / target_electron_density).max() < target_electron_density_max_residual,
                np.abs(change_in_target_electron_temp / target_electron_temp).max() < target_temp_max_residual,
            ]
        ):
            if not quiet:
                print(f"Converged in {iteration} iterations")
            break
    else:
        if two_point_model_error_nonconverged_error:
            raise RuntimeError("Iterative solve did not converge.")

    target_electron_flux_basic = calc_target_electron_flux_basic(**f_basic_kwargs)
    f_vol_loss_target_electron_flux = calc_f_vol_loss_target_electron_flux(**f_vol_loss_kwargs)

    target_electron_flux = calc_target_electron_flux(
        target_electron_flux_basic=target_electron_flux_basic,
        f_vol_loss_target_electron_flux=f_vol_loss_target_electron_flux,
        f_other_target_electron_flux=f_other_target_electron_flux,
    )

    upstream_SOL_collisionality = calc_upstream_SOL_collisionality(
        separatrix_electron_density=separatrix_electron_density,
        separatrix_electron_temp=separatrix_electron_temp,
        parallel_connection_length=parallel_connection_length,
    )

    kappa_e0 = Spitzer_conduction_reduction_factor * kappa_e0
    sheath_heat_transmission_factor = sheath_heat_transmission_factor + delta_electron_sheath_factor

    mask = (
        (np.abs(change_in_separatrix_electron_temp / separatrix_electron_temp) < upstream_temp_max_residual)
        & (np.abs(change_in_target_electron_density / target_electron_density) < target_electron_density_max_residual)
        & (np.abs(change_in_target_electron_temp / target_electron_temp) < target_temp_max_residual)
    )

    number_nonconverged = np.count_nonzero(~mask)
    if number_nonconverged > 0 and not quiet:
        print(f"{number_nonconverged} values did not converge in {max_iterations} iterations.")

    separatrix_electron_temp = xr.where(mask, separatrix_electron_temp, np.nan)  # type:ignore[no-untyped-call]
    target_electron_density = xr.where(mask, target_electron_density, np.nan)  # type:ignore[no-untyped-call]
    target_electron_temp = xr.where(mask, target_electron_temp, np.nan)  # type:ignore[no-untyped-call]
    target_electron_flux = xr.where(mask, target_electron_flux, np.nan)  # type:ignore[no-untyped-call]
    kappa_e0 = xr.where(mask, kappa_e0, np.nan)  # type:ignore[no-untyped-call]
    sheath_heat_transmission_factor = xr.where(mask, sheath_heat_transmission_factor, np.nan)  # type:ignore[no-untyped-call]
    Spitzer_conduction_reduction_factor = xr.where(mask, Spitzer_conduction_reduction_factor, np.nan)  # type:ignore[no-untyped-call]
    upstream_SOL_collisionality = xr.where(mask, upstream_SOL_collisionality, np.nan)  # type:ignore[no-untyped-call]
    delta_electron_sheath_factor = xr.where(mask, delta_electron_sheath_factor, np.nan)  # type:ignore[no-untyped-call]

    return (
        separatrix_electron_temp,
        target_electron_density,
        target_electron_temp,
        target_electron_flux,
        kappa_e0,
        sheath_heat_transmission_factor,
        Spitzer_conduction_reduction_factor,
        upstream_SOL_collisionality,
        delta_electron_sheath_factor,
    )
