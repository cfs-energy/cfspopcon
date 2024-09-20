"""Routines to calculate the target electron density, following the 2-point-model method of Stangeby, PPCF 2018."""

from typing import Union

import xarray as xr

from ....unit_handling import Unitfull


def calc_target_electron_density(
    target_electron_density_basic: Unitfull,
    f_vol_loss_target_electron_density: Union[float, xr.DataArray],
    f_other_target_electron_density: Union[float, xr.DataArray],
) -> Unitfull:
    """Calculate the target electron density, correcting for volume-losses and other effects.

    Components are calculated using the other functions in this file.
    """
    return target_electron_density_basic * f_vol_loss_target_electron_density * f_other_target_electron_density


def calc_target_electron_density_basic(
    average_ion_mass: Unitfull,
    q_parallel: Unitfull,
    upstream_total_pressure: Unitfull,
    sheath_heat_transmission_factor: Union[float, xr.DataArray],
) -> Unitfull:
    """Calculate the electron density at the target according to the basic two-point-model.

    From equation 24, :cite:`stangeby_2018`.

    Args:
        average_ion_mass: [amu]
        q_parallel: [GW/m^2]
        upstream_total_pressure: [atm]
        sheath_heat_transmission_factor: [~]

    Returns:
        target_electron_density_basic [m^-3]
    """
    return sheath_heat_transmission_factor**2 / (32.0 * average_ion_mass) * upstream_total_pressure**3 / q_parallel**2


def calc_f_vol_loss_target_electron_density(
    SOL_power_loss_fraction: Union[float, xr.DataArray],
    SOL_momentum_loss_fraction: Union[float, xr.DataArray],
) -> Union[float, xr.DataArray]:
    """Calculate the volume-loss correction term for the electron density at the target.

    From equation 24, :cite:`stangeby_2018`.

    Args:
        SOL_power_loss_fraction: f_cooling [~]
        SOL_momentum_loss_fraction: f_mom-loss [~]

    Returns:
        f_vol_loss_target_electron_density [~]
    """
    return (1.0 - SOL_momentum_loss_fraction) ** 3 / (1.0 - SOL_power_loss_fraction) ** 2


def calc_f_other_target_electron_density(
    target_ratio_of_ion_to_electron_temp: Union[float, xr.DataArray],
    target_ratio_of_electron_to_ion_density: Union[float, xr.DataArray],
    target_mach_number: Union[float, xr.DataArray],
    toroidal_flux_expansion: Union[float, xr.DataArray],
) -> Union[float, xr.DataArray]:
    """Calculate correction terms other than the volume-loss correction for the electron density at the target.

    Includes flux expansion, dilution of ions, different electron and ion temperatures and sub/super-sonic outflow.

    From equation 24, :cite:`stangeby_2018`., with

    Args:
        target_ratio_of_ion_to_electron_temp: tau_t = (T_i / T_e)_target [equation 21] [~]
        target_ratio_of_electron_to_ion_density: z_t = (ne / total ion density)_target [equation 22] [~]
        target_mach_number: M_t = (parallel ion velocity / sound speed)_target [~]
        toroidal_flux_expansion: R_t/R_u = major radius at target / major radius upstream [see discussion around equation 12] [~]

    Returns:
        f_other_target_electron_density [~]
    """
    return (
        (4.0 / (1.0 + target_ratio_of_ion_to_electron_temp / target_ratio_of_electron_to_ion_density) ** 2)
        * 8.0
        * target_mach_number**2
        / (1.0 + target_mach_number**2) ** 3
        * toroidal_flux_expansion**2
    )
