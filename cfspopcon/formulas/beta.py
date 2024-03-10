"""Calculate the ratio of magnetic to plasma (kinetic) pressure."""
import numpy as np

from ..unit_handling import Quantity, Unitfull, convert_units, ureg

def _calc_beta_general(
    average_electron_density: Unitfull, average_electron_temp: Unitfull, average_ion_temp: Unitfull, average_total_pressure: Unitfull, magnetic_field: Unitfull
) -> Unitfull:
    """Calculate the average ratio of the plasma pressure to the magnetic pressure due to a magnetic_field.

    Using equation 11.58 from Freidberg, "Plasma Physics and Fusion Energy" :cite:`freidberg_plasma_2007`

    The unit_conversion_factor comes from cancelling the units to get a dimensionless quantity

        >>> from pint import Quantity
        >>> n = Quantity(1e19, "m^-3")
        >>> T = Quantity(1, "keV")
        >>> B = Quantity(1, "T")
        >>> mu_0 = Quantity(1, "mu_0")
        >>> ((2*mu_0 * n * T / (B**2)).to('').units
        <Unit('dimensionless')>

    Args:
        average_electron_density: [1e19 m^-3] :term:`glossary link<average_electron_density>`
        average_electron_temp: [keV] :term:`glossary link<average_electron_temp>`
        average_ion_density: [1e19 m^-3] :term:`glossary link<average_ion_density>`
        average_ion_temp: [keV] :term:`glossary link<average_ion_temp>`
        average_total_pressure: [pascal] :term:`glossary link<average_total_pressure>`
        magnetic_field: magnetic field generating magnetic pressure [T]

    Returns:
         beta (toroidal or poloidal) [~]
    """
    mu_0 = Quantity(1, "mu_0")
    # to make the result dimensionless
    unit_conversion_factor = 2 * mu_0
    ret = unit_conversion_factor * average_total_pressure / (magnetic_field**2)
    return convert_units(ret, ureg.dimensionless)


def calc_beta_toroidal(
    average_electron_density: Unitfull, average_electron_temp: Unitfull, average_ion_temp: Unitfull, average_total_pressure: Unitfull, magnetic_field_on_axis: Unitfull,
ensity: Unitfull, average_ion_density: Unitfull
) -> Unitfull:
    """Calculate the average ratio of the plasma pressure to the magnetic pressure due to the toroidal field.

    Also called beta_external, since the toroidal field is generated by external toroidal field coils.
    Using equation 11.58 from Freidberg, "Plasma Physics and Fusion Energy" :cite:`freidberg_plasma_2007`

    Args:
        average_electron_density: [1e19 m^-3] :term:`glossary link<average_electron_density>`
        average_electron_temp: [keV] :term:`glossary link<average_electron_temp>`
        average_ion_density: [1e19 m^-3] :term:`glossary link<average_ion_density>`
        average_ion_temp: [keV] :term:`glossary link<average_ion_temp>`
        average_total_pressure: [pascal] :term:`glossary link<average_total_pressure>`
        magnetic_field_on_axis: [T] :term:`glossary link<magnetic_field_on_axis>`

    Returns:
         :term:`beta_toroidal` [~]
    """
    return _calc_beta_general(average_electron_density, average_electron_temp, average_ion_temp, average_total_pressure, magnetic_field=magnetic_field_on_axis)

def calc_beta_poloidal(
    average_electron_density: Unitfull,
    average_electron_temp: Unitfull,
    average_ion_density: Unitfull,
    average_ion_temp: Unitfull,
    plasma_current: Unitfull,
    average_total_pressure: Unitfull,
    minor_radius: Unitfull,
) -> Unitfull:
    """Calculate the average ratio of the plasma pressure to the magnetic pressure due to the plasma current.

    Calculates the poloidal magnetic field at radius a from the plasma current using
    equation 11.55 from Freidberg, "Plasma Physics and Fusion Energy" :cite:`freidberg_plasma_2007`
    and then evaluates beta_poloidal using
    equation 11.58 from Freidberg, "Plasma Physics and Fusion Energy" :cite:`freidberg_plasma_2007`

    The unit_conversion_factor cancels the units, and can be calculated using the following

        >>> from pint import Quantity
        >>> from numpy import pi
        >>> mu_0 = Quantity(1, "mu_0")
        >>> I = Quantity(1, "MA")
        >>> minor_radius = Quantity(1, "m")
        >>> (mu_0 * I / (2 * pi * minor_radius)).to("T").units
        <Unit('tesla')>

    Args:
        average_electron_density: [1e19 m^-3] :term:`glossary link<average_electron_density>`
        average_electron_temp: [keV] :term:`glossary link<average_electron_temp>`
        average_ion_density: [1e19 m^-3] :term:`glossary link<average_ion_density>`
        average_ion_temp: [keV] :term:`glossary link<average_ion_temp>`
        plasma_current: [MA] :term:`glossary link<plasma_current>`
        average_total_pressure: [pascal] :term:`glossary link<average_total_pressure>`
        minor_radius: [m] :term:`glossary link<minor_radius>`

    Returns:
        :term:`beta_poloidal` [~]
    """
    mu_0 = Quantity(1, "mu_0")
    # to ensure the final result is in units of tesla
    units_conversion_factor = mu_0 / (2 * np.pi)
    B_pol = units_conversion_factor * plasma_current / minor_radius

    return _calc_beta_general(average_electron_density, average_electron_temp, average_ion_temp, average_total_pressure, magnetic_field=B_pol)


def calc_beta_total(beta_toroidal: Unitfull, beta_poloidal: Unitfull) -> Unitfull:
    """Calculate the total beta from the toroidal and poloidal betas.

    Using equation 11.59 from Freidberg, "Plasma Physics and Fusion Energy" :cite:`freidberg_plasma_2007`

    Args:
        beta_toroidal: [~] :term:`glossary link<beta_toroidal>`
        beta_poloidal: [~] :term:`glossary link<beta_poloidal>`

    Returns:
         :term:`beta_total` [~]
    """
    return 1.0 / (1.0 / beta_toroidal + 1.0 / beta_poloidal)


def calc_beta_normalised(beta: Unitfull, minor_radius: Unitfull, magnetic_field_on_axis: Unitfull, plasma_current: Unitfull) -> Unitfull:
    """Normalize beta to stability (Troyon) parameters.

    See section 6.18 in Wesson :cite:`wesson_tokamaks_2011`.

    Args:
        beta: plasma pressure normalized against toroidal B-on-axis [%]
        minor_radius: [m] :term:`glossary link<minor_radius>`
        magnetic_field_on_axis: [T] :term:`glossary link<magnetic_field_on_axis>`
        plasma_current: [MA] :term:`glossary link<plasma_current>`

    Returns:
         :term:`beta_N`
    """
    normalisation = plasma_current / (minor_radius * magnetic_field_on_axis)

    beta_N = beta / normalisation

    return beta_N
