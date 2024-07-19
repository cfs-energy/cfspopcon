"""Compute all terms in the two-point-model for a fixed target electron temperature."""

from typing import Union

import xarray as xr

from ....named_options import MomentumLossFunction
from ....unit_handling import Unitfull, ureg
from ..separatrix_electron_temp import calc_separatrix_electron_temp
from .momentum_loss_functions import calc_SOL_momentum_loss_fraction
from .required_power_loss_fraction import calc_required_SOL_power_loss_fraction
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
from .target_electron_temp import calc_f_other_target_electron_temp, calc_target_electron_temp_basic


def solve_target_first_two_point_model(
    target_electron_temp: Unitfull,
    parallel_heat_flux_density: Unitfull,
    parallel_connection_length: Unitfull,
    separatrix_electron_density: Unitfull,
    toroidal_flux_expansion: Unitfull,
    fuel_average_mass_number: Unitfull,
    kappa_e0: Unitfull,
    SOL_momentum_loss_function: Union[MomentumLossFunction, xr.DataArray],
    sheath_heat_transmission_factor: Unitfull = 7.5 * ureg.dimensionless,
    SOL_conduction_fraction: Unitfull = 1.0 * ureg.dimensionless,
    target_ratio_of_ion_to_electron_temp: Unitfull = 1.0 * ureg.dimensionless,
    target_ratio_of_electron_to_ion_density: Unitfull = 1.0 * ureg.dimensionless,
    target_mach_number: Unitfull = 1.0 * ureg.dimensionless,
    upstream_ratio_of_ion_to_electron_temp: Unitfull = 1.0 * ureg.dimensionless,
    upstream_ratio_of_electron_to_ion_density: Unitfull = 1.0 * ureg.dimensionless,
    upstream_mach_number: Unitfull = 0.0 * ureg.dimensionless,
) -> tuple[Unitfull, Unitfull, Unitfull, Unitfull]:
    """Calculate the SOL_power_loss_fraction required to keep the target temperature at a given value.

    Args:
        target_electron_temp: [eV]
        parallel_heat_flux_density: [GW/m^2]
        parallel_connection_length: [m]
        separatrix_electron_density: [m^-3]
        toroidal_flux_expansion: [~]
        fuel_average_mass_number: [~]
        kappa_e0: electron heat conductivity constant [W / (eV^3.5 * m)]
        SOL_momentum_loss_function: which momentum loss function to use
        sheath_heat_transmission_factor: [~]
        SOL_conduction_fraction: [~]
        target_ratio_of_ion_to_electron_temp: [~]
        target_ratio_of_electron_to_ion_density: [~]
        target_mach_number: [~]
        upstream_ratio_of_ion_to_electron_temp: [~]
        upstream_ratio_of_electron_to_ion_density: [~]
        upstream_mach_number: [~]

    Returns:
        SOL_power_loss_fraction [~], separatrix_electron_temp [eV], target_electron_density [m^-3], target_electron_flux [m^-2 s^-1]
    """
    SOL_momentum_loss_fraction = calc_SOL_momentum_loss_fraction(SOL_momentum_loss_function, target_electron_temp)

    separatrix_electron_temp = calc_separatrix_electron_temp(
        target_electron_temp=target_electron_temp,
        parallel_heat_flux_density=parallel_heat_flux_density,
        parallel_connection_length=parallel_connection_length,
        SOL_conduction_fraction=SOL_conduction_fraction,
        kappa_e0=kappa_e0,
    )

    upstream_total_pressure = calc_upstream_total_pressure(
        separatrix_electron_density=separatrix_electron_density,
        separatrix_electron_temp=separatrix_electron_temp,
        upstream_ratio_of_ion_to_electron_temp=upstream_ratio_of_ion_to_electron_temp,
        upstream_ratio_of_electron_to_ion_density=upstream_ratio_of_electron_to_ion_density,
        upstream_mach_number=upstream_mach_number,
    )

    f_basic_kwargs = dict(
        fuel_average_mass_number=fuel_average_mass_number,
        parallel_heat_flux_density=parallel_heat_flux_density,
        upstream_total_pressure=upstream_total_pressure,
        sheath_heat_transmission_factor=sheath_heat_transmission_factor,
    )

    f_other_kwargs = dict(
        target_ratio_of_ion_to_electron_temp=target_ratio_of_ion_to_electron_temp,
        target_ratio_of_electron_to_ion_density=target_ratio_of_electron_to_ion_density,
        target_mach_number=target_mach_number,
        toroidal_flux_expansion=toroidal_flux_expansion,
    )

    SOL_power_loss_fraction = calc_required_SOL_power_loss_fraction(
        target_electron_temp_basic=calc_target_electron_temp_basic(**f_basic_kwargs),
        f_other_target_electron_temp=calc_f_other_target_electron_temp(**f_other_kwargs),
        SOL_momentum_loss_fraction=SOL_momentum_loss_fraction,
        required_target_electron_temp=target_electron_temp,
    )

    f_vol_loss_kwargs = dict(SOL_power_loss_fraction=SOL_power_loss_fraction, SOL_momentum_loss_fraction=SOL_momentum_loss_fraction)

    target_electron_density = calc_target_electron_density(
        target_electron_density_basic=calc_target_electron_density_basic(**f_basic_kwargs),
        f_vol_loss_target_electron_density=calc_f_vol_loss_target_electron_density(**f_vol_loss_kwargs),
        f_other_target_electron_density=calc_f_other_target_electron_density(**f_other_kwargs),
    )

    target_electron_flux = calc_target_electron_flux(
        target_electron_flux_basic=calc_target_electron_flux_basic(**f_basic_kwargs),
        f_vol_loss_target_electron_flux=calc_f_vol_loss_target_electron_flux(**f_vol_loss_kwargs),
        f_other_target_electron_flux=calc_f_other_target_electron_flux(**f_other_kwargs),
    )

    return SOL_power_loss_fraction, separatrix_electron_temp, target_electron_density, target_electron_flux
