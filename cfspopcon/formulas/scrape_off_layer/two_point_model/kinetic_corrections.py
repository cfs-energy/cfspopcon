"""Calculate quantities used in alternative parallel conduction models."""

from typing import Union

import numpy as np
import xarray as xr

from ....unit_handling import Unitfull, ureg, wraps_ufunc
from .separatrix_SOL_collisionality import calc_separatrix_SOL_collisionality


def calc_solkit_kinetic_corrections(
    separatrix_electron_density: Unitfull,
    target_electron_temp: Unitfull,
    separatrix_electron_temp: Unitfull,
    parallel_connection_length: Unitfull,
    SOL_momentum_loss_fraction: Unitfull,
) -> tuple[Unitfull, Unitfull, Unitfull]:
    """Calculate a consistent set of kinetic corrections, based on the SOL-KiT scalings.

    Args:
        separatrix_electron_density: [m^-3] :term:`glossary_link<separatrix_electron_density>`
        target_electron_temp: [eV] :term:`glossary_link<target_electron_temp>`
        separatrix_electron_temp: [eV] :term:`glossary_link<separatrix_electron_temp>`
        parallel_connection_length: [m] :term:`glossary_link<parallel_connection_length>`
        SOL_momentum_loss_fraction: [~] :term:`glossary_link<SOL_momentum_loss_fraction>`

    Returns:
        :term:`separatrix_SOL_collisionality` [~], :term:`Spitzer_conduction_reduction_factor` [~], :term:`delta_electron_sheath_factor` [~]
    """
    separatrix_SOL_collisionality = calc_separatrix_SOL_collisionality(
        separatrix_electron_density=separatrix_electron_density,
        separatrix_electron_temp=separatrix_electron_temp,
        parallel_connection_length=parallel_connection_length,
    )

    Spitzer_conduction_reduction_factor = calc_Spitzer_conduction_reduction_factor_scaling(
        separatrix_SOL_collisionality=separatrix_SOL_collisionality,
    )

    delta_electron_sheath_factor = calc_delta_electron_sheath_factor(
        separatrix_electron_temp=separatrix_electron_temp,
        target_electron_temp=target_electron_temp,
        SOL_momentum_loss_fraction=SOL_momentum_loss_fraction,
    )
    return separatrix_SOL_collisionality, Spitzer_conduction_reduction_factor, delta_electron_sheath_factor


def calc_Spitzer_conduction_reduction_factor_scaling(
    separatrix_SOL_collisionality: Unitfull,
) -> Union[float, xr.DataArray]:
    """Factor to reduce the electron heat conduction (relative to Braginskii) due to kinetic effects, according to SOL-KiT.

    Equation 10 from :cite:`Power_2023`

    Args:
        separatrix_SOL_collisionality: [~] :term:`glossary_link<separatrix_SOL_collisionality>`

    Returns:
        :term:`Spitzer_conduction_reduction_factor` [~]
    """
    Spitzer_conduction_reduction_factor = 0.696 * np.exp(-8.059 * (separatrix_SOL_collisionality) ** -1.074) + 0.260
    Spitzer_conduction_reduction_factor = xr.where(
        Spitzer_conduction_reduction_factor < 0.5, 1.0 / (1 + 6.584 / separatrix_SOL_collisionality), Spitzer_conduction_reduction_factor
    )

    return Spitzer_conduction_reduction_factor


@wraps_ufunc(
    return_units=dict(Spitzer_conduction_reduction_factor=ureg.dimensionless),
    input_units=dict(
        separatrix_electron_density=ureg.m**-3,
        separatrix_electron_temp=ureg.eV,
        parallel_connection_length=ureg.m,
        target_electron_temp=ureg.eV,
        kappa_e0=ureg.W / (ureg.eV**3.5 * ureg.m),
        electron_mass=ureg.kg,  # TODO: electron mass in kg?
        electron_charge=ureg.C,  # TODO: elementary charge in Coulomb?
        SOL_conduction_fraction=ureg.dimensionless,
        flux_limit_factor_alpha=ureg.dimensionless,
    ),
)
def calc_Spitzer_conduction_reduction_factor_fluxlim(
    separatrix_electron_density: Unitfull,
    separatrix_electron_temp: Unitfull,
    parallel_connection_length: Unitfull,
    target_electron_temp: Unitfull,
    kappa_e0: Unitfull,
    electron_mass: Unitfull = 1.0 * ureg.electron_mass,
    electron_charge: Unitfull = 1.0 * ureg.elementary_charge,
    SOL_conduction_fraction: Union[float, xr.DataArray] = 1.0,
    flux_limit_factor_alpha: Union[float, xr.DataArray] = 0.15,
) -> Union[float, xr.DataArray]:
    """Factor to reduce the electron heat conduction (relative to Braginskii) due to kinetic effects, via a flux limiter.

    #TODO: Needs reference and units need to be checked

    Args:
        separatrix_electron_density: [m^-3] :term:`glossary_link<separatrix_electron_density>`
        separatrix_electron_temp: [eV] :term:`glossary_link<separatrix_electron_temp>`
        parallel_connection_length: [m] :term:`glossary_link<parallel_connection_length>`
        target_electron_temp: [eV] :term:`glossary_link<target_electron_temp>`
        kappa_e0: [W / (eV^3.5 * m)] :term:`glossary_link<kappa_e0>`
        electron_mass: [kg] :term:`glossary_link<electron_mass>`
        electron_charge: [C] :term:`glossary_link<electron_charge>`
        SOL_conduction_fraction: [~] :term:`glossary_link<SOL_conduction_fraction>`
        flux_limit_factor_alpha: [~] :term:`glossary_link<flux_limit_factor_alpha>`

    Returns:
        :term:`Spitzer_conduction_reduction_factor` [~]
    """
    spitzer_heat_flux = (
        (2.0 / 7)
        * (kappa_e0 / (parallel_connection_length * SOL_conduction_fraction))
        * ((separatrix_electron_temp) ** 3.5 - (target_electron_temp) ** 3.5)
    )
    free_streaming_heat_flux = (
        separatrix_electron_density * (separatrix_electron_temp * electron_charge) ** 1.5 * (1.0 / electron_mass) ** 0.5
    )
    Spitzer_conduction_reduction_factor = 1 / (1 + spitzer_heat_flux / (flux_limit_factor_alpha * free_streaming_heat_flux))

    return Spitzer_conduction_reduction_factor


def calc_delta_electron_sheath_factor(
    separatrix_electron_temp: Unitfull,
    target_electron_temp: Unitfull,
    SOL_momentum_loss_fraction: Union[float, xr.DataArray] = 0.0,
) -> Union[float, xr.DataArray]:
    """Offset to increase the sheath heat transmission factor (relative to fluid) due to kinetic effects, according to SOL-KiT.

    #TODO: needs reference

    Args:
        separatrix_electron_temp: [n19] :term:`glossary_link<separatrix_electron_temp>`
        target_electron_temp: [eV] :term:`glossary_link<target_electron_temp>`
        SOL_momentum_loss_fraction: [~] :term:`glossary_link<SOL_momentum_loss_fraction>`

    Returns:
        :term:`delta_electron_sheath_factor:
    """
    delta_electron_sheath_factor = 1.08 * (1 - SOL_momentum_loss_fraction) * (separatrix_electron_temp / target_electron_temp) ** 0.25

    return delta_electron_sheath_factor
