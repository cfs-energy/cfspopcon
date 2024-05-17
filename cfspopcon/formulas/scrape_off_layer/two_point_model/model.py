"""Compute all terms in the two-point-model for a fixed SOL power loss fraction."""
from typing import Union

import numpy as np
import xarray as xr

from ....named_options import MomentumLossFunction
from ....unit_handling import Quantity, Unitfull, ureg
from ..separatrix_electron_temp import calc_separatrix_electron_temp
from .momentum_loss_functions import calc_SOL_momentum_loss_fraction
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


def solve_two_point_model(
    SOL_power_loss_fraction: Unitfull,
    parallel_heat_flux_density: Unitfull,
    parallel_connection_length: Unitfull,
    separatrix_electron_density: Unitfull,
    toroidal_flux_expansion: Unitfull,
    fuel_average_mass_number: Unitfull,
    kappa_e0: Unitfull,
    SOL_momentum_loss_function: Union[MomentumLossFunction, xr.DataArray],
    initial_target_electron_temp: Unitfull = 10.0 * ureg.eV,
    sheath_heat_transmission_factor: Unitfull = 7.5 * ureg.dimensionless,
    SOL_conduction_fraction: Unitfull = 1.0 * ureg.dimensionless,
    target_ratio_of_ion_to_electron_temp: Unitfull = 1.0 * ureg.dimensionless,
    target_ratio_of_electron_to_ion_density: Unitfull = 1.0 * ureg.dimensionless,
    target_mach_number: Unitfull = 1.0 * ureg.dimensionless,
    upstream_ratio_of_ion_to_electron_temp: Unitfull = 1.0 * ureg.dimensionless,
    upstream_ratio_of_electron_to_ion_density: Unitfull = 1.0 * ureg.dimensionless,
    upstream_mach_number: Unitfull = 0.0 * ureg.dimensionless,
    # Controlling the iterative solve
    max_iterations: int = 100,
    upstream_temp_relaxation: float = 0.5,
    target_electron_density_relaxation: float = 0.5,
    target_temp_relaxation: float = 0.5,
    upstream_temp_max_residual: float = 1e-2,
    target_electron_density_max_residual: float = 1e-2,
    target_temp_max_residual: float = 1e-2,
    # Return converged values even if not whole array is converged
    raise_error_if_not_converged: bool = True,
    # Print information about the solve to terminal
    quiet: bool = True,
) -> tuple[Union[Quantity, xr.DataArray], Union[Quantity, xr.DataArray], Union[Quantity, xr.DataArray], Union[Quantity, xr.DataArray]]:
    """Calculate the upstream and target electron temperature and target electron density according to the extended two-point-model.

    Args:
        SOL_power_loss_fraction: [~]
        parallel_heat_flux_density: [GW/m^2]
        parallel_connection_length: [m]
        separatrix_electron_density: [m^-3]
        toroidal_flux_expansion: [~]
        fuel_average_mass_number: [~]
        kappa_e0: electron heat conductivity constant [W / (eV^3.5 * m)]
        SOL_momentum_loss_function: which momentum loss function to use
        initial_target_electron_temp: starting guess for target electron temp [eV]
        sheath_heat_transmission_factor: [~]
        SOL_conduction_fraction: [~]
        target_ratio_of_ion_to_electron_temp: [~]
        target_ratio_of_electron_to_ion_density: [~]
        target_mach_number: [~]
        upstream_ratio_of_ion_to_electron_temp: [~]
        upstream_ratio_of_electron_to_ion_density: [~]
        upstream_mach_number: [~]
        max_iterations: how many iterations to try before returning NaN
        upstream_temp_relaxation: step-size for upstream Te evolution
        target_electron_density_relaxation: step-size for target ne evolution
        target_temp_relaxation: step-size for target Te evolution
        upstream_temp_max_residual: relative rate of change for convergence for upstream Te evolution
        target_electron_density_max_residual: relative rate of change for convergence for target ne evolution
        target_temp_max_residual: relative rate of change for convergence for target Te evolution
        raise_error_if_not_converged: raise an error if not all point converge within max iterations (otherwise return NaN)
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

    while iteration < max_iterations:
        iteration += 1

        new_separatrix_electron_temp = calc_separatrix_electron_temp(
            target_electron_temp=target_electron_temp,
            parallel_heat_flux_density=parallel_heat_flux_density,
            parallel_connection_length=parallel_connection_length,
            SOL_conduction_fraction=SOL_conduction_fraction,
            kappa_e0=kappa_e0,
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
            fuel_average_mass_number=fuel_average_mass_number,
            parallel_heat_flux_density=parallel_heat_flux_density,
            upstream_total_pressure=upstream_total_pressure,
            sheath_heat_transmission_factor=sheath_heat_transmission_factor,
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
        if raise_error_if_not_converged:
            raise RuntimeError("Iterative solve did not converge.")

    target_electron_flux_basic = calc_target_electron_flux_basic(**f_basic_kwargs)
    f_vol_loss_target_electron_flux = calc_f_vol_loss_target_electron_flux(**f_vol_loss_kwargs)

    target_electron_flux = calc_target_electron_flux(
        target_electron_flux_basic=target_electron_flux_basic,
        f_vol_loss_target_electron_flux=f_vol_loss_target_electron_flux,
        f_other_target_electron_flux=f_other_target_electron_flux,
    )

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

    return separatrix_electron_temp, target_electron_density, target_electron_temp, target_electron_flux
