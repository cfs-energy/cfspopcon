"""Solve the Lengyel model for a consistent (cz, Tu) pair."""

from typing import Callable

import numpy as np
import xarray as xr
from scipy.interpolate import InterpolatedUnivariateSpline  # type:ignore[import-untyped]

from ...named_options import AtomicSpecies, MomentumLossFunction
from ...unit_handling import Unitfull, convert_units, ureg, wraps_ufunc
from ..scrape_off_layer.two_point_model.target_first_model import (  # type:ignore[attr-defined]
    calc_f_other_target_electron_temp,
    calc_required_SOL_power_loss_fraction,
    calc_separatrix_electron_temp,
    calc_SOL_momentum_loss_fraction,
    calc_target_electron_temp_basic,
    calc_upstream_total_pressure,
)
from .edge_radiator_conc import build_L_int_integrator, calc_required_edge_impurity_concentration


def calc_full_lengyel_model(
    upstream_parallel_heat_flux_density: Unitfull,
    parallel_connection_length: Unitfull,
    fuel_average_mass_number: Unitfull,
    edge_impurity_species: AtomicSpecies,
    target_electron_temp: Unitfull,
    separatrix_electron_density: Unitfull,
    kappa_e0: Unitfull,
    atomic_data: xr.DataArray,
    reference_electron_density: Unitfull = 1.0 * ureg.n20,
    reference_ne_tau: Unitfull = 1.0 * ureg.n20 * ureg.ms,
    SOL_conduction_fraction: Unitfull = 1.0,
    target_ratio_of_ion_to_electron_temp: Unitfull = 1.0,
    target_ratio_of_electron_to_ion_density: Unitfull = 1.0,
    target_mach_number: Unitfull = 1.0,
    toroidal_flux_expansion: Unitfull = 1.0,
    upstream_ratio_of_ion_to_electron_temp: Unitfull = 1.0,
    upstream_ratio_of_electron_to_ion_density: Unitfull = 1.0,
    upstream_mach_number: Unitfull = 0.0,
    sheath_heat_transmission_factor: Unitfull = 7.5,
    iterations: int = 5,
) -> tuple[Unitfull, Unitfull, Unitfull]:
    """Compute a consistent cz, Tu pair using the Lengyel model, for a fixed target electron temperature.

    Args:
        upstream_parallel_heat_flux_density: :term:`glossary link<upstream_parallel_heat_flux_density>`
        parallel_connection_length: :term:`glossary link<parallel_connection_length>`
        fuel_average_mass_number: :term:`glossary link<fuel_average_mass_number>`
        edge_impurity_species: :term:`glossary link<edge_impurity_species>`
        target_electron_temp: :term:`glossary link<target_electron_temp>`
        separatrix_electron_density: :term:`glossary link<separatrix_electron_density>`
        kappa_e0: :term:`glossary link<kappa_e0>`
        atomic_data: :term:`glossary link<atomic_data>`
        reference_electron_density: :term:`glossary link<reference_electron_density>`
        reference_ne_tau: :term:`glossary link<reference_ne_tau>`
        SOL_conduction_fraction: :term:`glossary link<SOL_conduction_fraction>`
        target_ratio_of_ion_to_electron_temp: :term:`glossary link<target_ratio_of_ion_to_electron_temp>`
        target_ratio_of_electron_to_ion_density: :term:`glossary link<target_ratio_of_electron_to_ion_density>`
        target_mach_number: :term:`glossary link<target_mach_number>`
        toroidal_flux_expansion: :term:`glossary link<toroidal_flux_expansion>`
        upstream_ratio_of_ion_to_electron_temp: :term:`glossary link<upstream_ratio_of_ion_to_electron_temp>`
        upstream_ratio_of_electron_to_ion_density: :term:`glossary link<upstream_ratio_of_electron_to_ion_density>`
        upstream_mach_number: :term:`glossary link<upstream_mach_number>`
        sheath_heat_transmission_factor: :term:`glossary link<sheath_heat_transmission_factor>`
        iterations: number of iterations to find consistent cz, Tu pair

    Returns:
        :term:`edge_impurity_concentration`, :term:`SOL_power_loss_fraction`, :term:`separatrix_electron_temp`
    """
    L_int_integrator = build_L_int_integrator(
        atomic_data=atomic_data.item() if isinstance(atomic_data, xr.DataArray) else atomic_data,
        impurity_species=edge_impurity_species,
        reference_electron_density=reference_electron_density,
        reference_ne_tau=reference_ne_tau,
    )

    SOL_momentum_loss_fraction = calc_SOL_momentum_loss_fraction(MomentumLossFunction.KotovReiter, target_electron_temp)

    separatrix_electron_temp_spitzer_harm = calc_separatrix_electron_temp(
        target_electron_temp=target_electron_temp,
        parallel_heat_flux_density=upstream_parallel_heat_flux_density,
        parallel_connection_length=parallel_connection_length,
        SOL_conduction_fraction=SOL_conduction_fraction,
        kappa_e0=kappa_e0,
    )
    separatrix_electron_temp = separatrix_electron_temp_spitzer_harm

    f_other_target_electron_temp = calc_f_other_target_electron_temp(
        target_ratio_of_ion_to_electron_temp=target_ratio_of_ion_to_electron_temp,
        target_ratio_of_electron_to_ion_density=target_ratio_of_electron_to_ion_density,
        target_mach_number=target_mach_number,
        toroidal_flux_expansion=toroidal_flux_expansion,
    )

    def upstream_total_pressure(separatrix_electron_temp: Unitfull) -> Unitfull:
        return calc_upstream_total_pressure(
            separatrix_electron_density=separatrix_electron_density,
            separatrix_electron_temp=separatrix_electron_temp,
            upstream_ratio_of_ion_to_electron_temp=upstream_ratio_of_ion_to_electron_temp,
            upstream_ratio_of_electron_to_ion_density=upstream_ratio_of_electron_to_ion_density,
            upstream_mach_number=upstream_mach_number,
        )

    def target_electron_temp_basic(separatrix_electron_temp: Unitfull) -> Unitfull:
        return calc_target_electron_temp_basic(
            parallel_heat_flux_density=upstream_parallel_heat_flux_density,
            fuel_average_mass_number=fuel_average_mass_number,
            upstream_total_pressure=upstream_total_pressure(separatrix_electron_temp),
            sheath_heat_transmission_factor=sheath_heat_transmission_factor,
        )

    for _ in range(iterations):

        SOL_power_loss_fraction = calc_required_SOL_power_loss_fraction(
            target_electron_temp_basic=target_electron_temp_basic(separatrix_electron_temp),
            f_other_target_electron_temp=f_other_target_electron_temp,
            SOL_momentum_loss_fraction=SOL_momentum_loss_fraction,
            required_target_electron_temp=target_electron_temp,
        )

        edge_impurity_concentration = convert_units(
            calc_required_edge_impurity_concentration(
                L_int_integrator=L_int_integrator,
                q_parallel=upstream_parallel_heat_flux_density,
                SOL_power_loss_fraction=SOL_power_loss_fraction,
                target_electron_temp=target_electron_temp,
                separatrix_electron_temp=separatrix_electron_temp,
                separatrix_electron_density=separatrix_electron_density,
                kappa_e0=kappa_e0,
                lengyel_overestimation_factor=1.0,
            ),
            ureg.dimensionless,
        )

        separatrix_electron_temp = calc_separatrix_electron_temp_from_lengyel(
            parallel_connection_length=parallel_connection_length,
            target_electron_temp=target_electron_temp,
            separatrix_electron_temp_spitzer_harm=separatrix_electron_temp_spitzer_harm,
            upstream_q_parallel=upstream_parallel_heat_flux_density,
            radiated_fraction=SOL_power_loss_fraction,
            upstream_total_pressure=upstream_total_pressure(separatrix_electron_temp),
            impurity_fraction=edge_impurity_concentration,
            kappa_e0=kappa_e0,
            L_int_integrator=L_int_integrator,
        )

    return edge_impurity_concentration, SOL_power_loss_fraction, separatrix_electron_temp


@wraps_ufunc(
    input_units=dict(
        parallel_connection_length=ureg.m,
        target_electron_temp=ureg.eV,
        separatrix_electron_temp_spitzer_harm=ureg.eV,
        upstream_q_parallel=ureg.W / ureg.m**2,
        radiated_fraction=ureg.dimensionless,
        upstream_total_pressure=ureg.eV / ureg.m**3,
        impurity_fraction=ureg.dimensionless,
        kappa_e0=ureg.W / (ureg.eV**3.5 * ureg.m),
        L_int_integrator=None,
    ),
    return_units=dict(separatrix_electron_temp=ureg.eV),
    pass_as_kwargs=("L_int_integrator",),
)
def calc_separatrix_electron_temp_from_lengyel(
    parallel_connection_length: float,
    target_electron_temp: float,
    separatrix_electron_temp_spitzer_harm: float,
    upstream_q_parallel: float,
    radiated_fraction: float,
    upstream_total_pressure: float,
    impurity_fraction: float,
    kappa_e0: float,
    L_int_integrator: Callable[[Unitfull, Unitfull], Unitfull],
) -> float:
    """Calculate a value for the upstream temperature which is consistent with the parallel connection length."""
    iterations = 5
    factor_Te_sep = 5.0
    max_temp = separatrix_electron_temp_spitzer_harm * factor_Te_sep

    electron_temp = np.linspace(target_electron_temp, max_temp)
    electron_temp = electron_temp[1:]

    L_par_integrand = np.zeros_like(electron_temp)

    for i, Te in enumerate(electron_temp):
        L_int = np.maximum(L_int_integrator.unitless_func(target_electron_temp, Te), 0.0)  # type:ignore[attr-defined]

        q_par_target_squared = np.power((1.0 - radiated_fraction) * upstream_q_parallel, 2)
        delta_qpar_squared = 2.0 * kappa_e0 * np.power(upstream_total_pressure, 2) * impurity_fraction * L_int

        assert (q_par_target_squared + delta_qpar_squared) > 0
        q_par_Te = np.sqrt(q_par_target_squared + delta_qpar_squared)

        L_par_integrand[i] = kappa_e0 * np.power(Te, 2.5) / q_par_Te

    interpolator = InterpolatedUnivariateSpline(electron_temp, L_par_integrand)

    low_temp, high_temp = target_electron_temp, max_temp
    for _ in range(iterations):
        separatrix_electron_temp = (low_temp + high_temp) / 2.0
        L_par_test = interpolator.integral(target_electron_temp, separatrix_electron_temp)

        if L_par_test < parallel_connection_length:
            low_temp = separatrix_electron_temp
        else:
            high_temp = separatrix_electron_temp

    assert (low_temp > target_electron_temp) and (high_temp < max_temp)

    return separatrix_electron_temp
