"""Routines to calculate the target electron flux, following the 2-point-model method of Stangeby, PPCF 2018."""

from typing import Union

import xarray as xr

from ....unit_handling import Quantity


def calc_target_electron_flux(
    target_electron_flux_basic: Union[Quantity, xr.DataArray],
    f_vol_loss_target_electron_flux: Union[float, xr.DataArray],
    f_other_target_electron_flux: Union[float, xr.DataArray],
) -> Union[Quantity, xr.DataArray]:
    """Calculate the target electron flux, correcting for volume-losses and other effects.

    Components are calculated using the other functions in this file.
    """
    return (
        target_electron_flux_basic
        * f_vol_loss_target_electron_flux
        * f_other_target_electron_flux
    )


def calc_target_electron_flux_basic(
    fuel_average_mass_number: Union[Quantity, xr.DataArray],
    parallel_heat_flux_density: Union[Quantity, xr.DataArray],
    upstream_total_pressure: Union[Quantity, xr.DataArray],
    sheath_heat_transmission_factor: Union[float, xr.DataArray],
) -> Union[Quantity, xr.DataArray]:
    """Calculate the flux of electrons (particles per square-metre per second) at the target according to the basic two-point-model.

    From equation 24, :cite:`stangeby_2018`.

    Args:
        fuel_average_mass_number: [amu]
        parallel_heat_flux_density: [GW/m^2]
        upstream_total_pressure: [atm]
        sheath_heat_transmission_factor: [~]

    Returns:
        target_electron_flux_basic [m^-2 s^-1]
    """
    return (
        sheath_heat_transmission_factor
        / (8.0 * fuel_average_mass_number)
        * upstream_total_pressure**2
        / parallel_heat_flux_density
    )


def calc_f_vol_loss_target_electron_flux(
    SOL_power_loss_fraction: Union[float, xr.DataArray],
    SOL_momentum_loss_fraction: Union[float, xr.DataArray],
) -> Union[float, xr.DataArray]:
    """Calculate the volume-loss correction term for the electron flux at the target.

    From equation 24, :cite:`stangeby_2018`.

    Args:
        SOL_power_loss_fraction: f_cooling [~]
        SOL_momentum_loss_fraction: f_mom-loss [~]

    Returns:
        f_vol_loss_target_electron_flux [~]
    """
    return (1.0 - SOL_momentum_loss_fraction) ** 2 / (1.0 - SOL_power_loss_fraction)


def calc_f_other_target_electron_flux(
    target_ratio_of_ion_to_electron_temp: Union[float, xr.DataArray],
    target_ratio_of_electron_to_ion_density: Union[float, xr.DataArray],
    target_mach_number: Union[float, xr.DataArray],
    toroidal_flux_expansion: Union[float, xr.DataArray],
) -> Union[float, xr.DataArray]:
    """Calculate correction terms other than the volume-loss correction for the electron flux at the target.

    Includes flux expansion, dilution of ions, different electron and ion temperatures and sub/super-sonic outflow.
    From equation 24, :cite:`stangeby_2018`., with

    Args:
        target_ratio_of_ion_to_electron_temp: tau_t = (T_i / T_e)_target [equation 21] [~]
        target_ratio_of_electron_to_ion_density: z_t = (ne / total ion density)_target [equation 22] [~]
        target_mach_number: M_t = (parallel ion velocity / sound speed)_target [~]
        toroidal_flux_expansion: R_t/R_u = major radius at target / major radius upstream [see discussion around equation 12] [~]

    Returns:
        f_other_target_electron_flux [~]
    """
    return (
        (
            2.0
            / (
                1.0
                + target_ratio_of_ion_to_electron_temp
                / target_ratio_of_electron_to_ion_density
            )
        )
        * 4.0
        * target_mach_number**2
        / (1.0 + target_mach_number**2) ** 2
        * toroidal_flux_expansion
    )
