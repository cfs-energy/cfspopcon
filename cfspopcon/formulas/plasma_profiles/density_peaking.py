"""Estimate the density peaking based on scaling from C. Angioni."""

import numpy as np
import xarray as xr

from ...algorithm_class import Algorithm
from ...unit_handling import Unitfull, ureg, wraps_ufunc


def calc_density_peaking(
    effective_collisionality: Unitfull, beta_toroidal: Unitfull, nu_noffset: Unitfull
) -> Unitfull:
    """Calculate the density peaking (peak over volume average).

    Equation 3 from p1334 of Angioni et al, "Scaling of density peaking in H-mode plasmas based on a combined
    database of AUG and JET observations" :cite:`angioni_scaling_2007`

    Args:
        effective_collisionality: [~] :term:`glossary link <effective_collisionality>`
        beta_toroidal: :term:`glossary link <beta_toroidal>` [~]
        nu_noffset: scalar offset added to peaking factor [~]

    Returns:
        :term:`nu_n` [~]
    """
    nu_n = (
        1.347 - 0.117 * np.log(effective_collisionality) - 4.03 * beta_toroidal
    ) + nu_noffset
    if isinstance(nu_n, xr.DataArray):
        return nu_n.clip(1.0, float("inf"))
    else:
        return max(nu_n, 1.0 * ureg.dimensionless)


@Algorithm.register_algorithm(
    return_keys=["ion_density_peaking", "peak_fuel_ion_density"]
)
def calc_ion_density_peaking(
    effective_collisionality: Unitfull,
    beta_toroidal: Unitfull,
    ion_density_peaking_offset: Unitfull,
    average_electron_density: Unitfull,
    dilution: Unitfull,
) -> tuple[Unitfull, ...]:
    """Calculate the ion density peaking.

    Args:
        effective_collisionality: [~] :term:`glossary link <effective_collisionality>`
        beta_toroidal: :term:`glossary link <beta_toroidal>` [~]
        ion_density_peaking_offset: :term:`glossary link<ion_density_peaking_offset>` [~]
        average_electron_density: :term:`glossary link<average_electron_density>`
        dilution: :term:`glossary link<dilution>`

    Returns:
        :term:`ion_density_peaking`, :term:`peak_fuel_ion_density`
    """
    ion_density_peaking = calc_density_peaking(
        effective_collisionality, beta_toroidal, nu_noffset=ion_density_peaking_offset
    )
    peak_fuel_ion_density = average_electron_density * dilution * ion_density_peaking

    return ion_density_peaking, peak_fuel_ion_density


@Algorithm.register_algorithm(
    return_keys=["electron_density_peaking", "peak_electron_density"]
)
def calc_electron_density_peaking(
    effective_collisionality: Unitfull,
    beta_toroidal: Unitfull,
    electron_density_peaking_offset: Unitfull,
    average_electron_density: Unitfull,
) -> tuple[Unitfull, ...]:
    """Calculate the electron density peaking.

    Args:
        effective_collisionality: [~] :term:`glossary link <effective_collisionality>`
        beta_toroidal: :term:`glossary link <beta_toroidal>` [~]
        electron_density_peaking_offset: :term:`glossary link<electron_density_peaking_offset>` [~]
        average_electron_density: :term:`glossary link<average_electron_density>`

    Returns:
        :term:`electron_density_peaking`, :term:`peak_electron_density`
    """
    electron_density_peaking = calc_density_peaking(
        effective_collisionality,
        beta_toroidal,
        nu_noffset=electron_density_peaking_offset,
    )
    peak_electron_density = average_electron_density * electron_density_peaking

    return electron_density_peaking, peak_electron_density


@Algorithm.register_algorithm(return_keys=["effective_collisionality"])
@wraps_ufunc(
    return_units=dict(effective_collisionality=ureg.dimensionless),
    input_units=dict(
        average_electron_density=ureg.n19,
        average_electron_temp=ureg.keV,
        major_radius=ureg.m,
        z_effective=ureg.dimensionless,
    ),
)
def calc_effective_collisionality(
    average_electron_density: float,
    average_electron_temp: float,
    major_radius: float,
    z_effective: float,
) -> float:
    """Calculate the effective collisionality.

    From p1327 of Angioni et al, "Scaling of density peaking in H-mode plasmas based on a combined
    database of AUG and JET observations" :cite:`angioni_scaling_2007`

    Args:
        average_electron_density: [1e19 m^-3] :term:`glossary link<average_electron_density>`
        average_electron_temp: [keV] :term:`glossary link<average_electron_temp>`
        major_radius: [m] :term:`glossary link<major_radius>`
        z_effective: [~] :term:`glossary link<z_effective>`

    Returns:
        :term:`effective_collisionality` [~]
    """
    return float(
        (0.1 * z_effective * average_electron_density * major_radius)
        / (average_electron_temp**2.0)
    )
