"""Routines to calculate the combined electron and ion pressure in the SOL."""


import xarray as xr

from ....unit_handling import Quantity


def calc_upstream_total_pressure(
    separatrix_electron_density: Quantity | xr.DataArray,
    separatrix_electron_temp: Quantity | xr.DataArray,
    upstream_ratio_of_ion_to_electron_temp: float | xr.DataArray,
    upstream_ratio_of_electron_to_ion_density: float | xr.DataArray,
    upstream_mach_number: float | xr.DataArray = 0.0,
) -> Quantity | xr.DataArray:
    """Calculate the upstream total pressure (including the ion temperature contribution).

    Same as calc_total_pressure, but with a default value upstream_mach_number=0.0.

    Args:
        separatrix_electron_density: [m^-3]
        separatrix_electron_temp: [eV]
        upstream_ratio_of_ion_to_electron_temp: tau_t = (T_i / T_e) [~]
        upstream_ratio_of_electron_to_ion_density: z_t = (ne / ni) [~]
        upstream_mach_number: M_t = (parallel ion velocity / sound speed) [~]

    Returns:
        upstream_total_pressure [atm]
    """
    return calc_total_pressure(
        electron_density=separatrix_electron_density,
        electron_temp=separatrix_electron_temp,
        ratio_of_ion_to_electron_temp=upstream_ratio_of_ion_to_electron_temp,
        ratio_of_electron_to_ion_density=upstream_ratio_of_electron_to_ion_density,
        mach_number=upstream_mach_number,
    )


def calc_total_pressure(
    electron_density: Quantity | xr.DataArray,
    electron_temp: Quantity | xr.DataArray,
    ratio_of_ion_to_electron_temp: float | xr.DataArray,
    ratio_of_electron_to_ion_density: float | xr.DataArray,
    mach_number: float | xr.DataArray,
) -> Quantity | xr.DataArray:
    """Calculate the total pressure (including ion temperature contribution).

    From equation 20, :cite:`stangeby_2018`.

    Args:
        electron_density: [m^-3]
        electron_temp: [eV]
        ratio_of_ion_to_electron_temp: tau_t = (T_i / T_e) [~]
        ratio_of_electron_to_ion_density: z_t = (ne / ni) [~]
        mach_number: M_t = (parallel ion velocity / sound speed) [~]

    Returns:
        upstream_total_pressure [atm]
    """
    return (
        (1.0 + mach_number**2) * electron_density * electron_temp * (1.0 + ratio_of_ion_to_electron_temp / ratio_of_electron_to_ion_density)
    )
