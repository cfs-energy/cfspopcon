"""Estimate the density peaking based on scaling from C. Angioni."""
import numpy as np
import xarray as xr

from ..unit_handling import Unitfull, ureg, wraps_ufunc


def calc_density_peaking(effective_collisionality: Unitfull, betaE: Unitfull, nu_noffset: Unitfull) -> Unitfull:
    """Calculate the density peaking (peak over volume average).

    Equation 3 from p1334 of Angioni et al, "Scaling of density peaking in H-mode plasmas based on a combined
    database of AUG and JET observations" :cite:`angioni_scaling_2007`

    Args:
        effective_collisionality: [~] :term:`glossary link <effective_collisionality>`
        betaE: beta due to external field [~]
        nu_noffset: scalar offset added to peaking factor [~]

    Returns:
        :term:`nu_n` [~]
    """
    nu_n = (1.347 - 0.117 * np.log(effective_collisionality) - 4.03 * betaE) + nu_noffset
    if isinstance(nu_n, xr.DataArray):
        return nu_n.clip(1.0, float("inf"))
    else:
        return max(nu_n, 1.0 * ureg.dimensionless)


@wraps_ufunc(
    return_units=dict(effective_collisionality=ureg.dimensionless),
    input_units=dict(
        average_electron_density=ureg.n19, average_electron_temp=ureg.keV, major_radius=ureg.m, z_effective=ureg.dimensionless
    ),
)
def calc_effective_collisionality(
    average_electron_density: float, average_electron_temp: float, major_radius: float, z_effective: float
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
    return float((0.1 * z_effective * average_electron_density * major_radius) / (average_electron_temp**2.0))
