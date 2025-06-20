"""Lengyel model to compute the edge impurity concentration."""

from typing import Callable

import numpy as np
import xarray as xr
from scipy.interpolate import InterpolatedUnivariateSpline

from ...algorithm_class import Algorithm
from ...helpers import get_item
from ...named_options import AtomicSpecies
from ...unit_handling import Unitfull, magnitude, magnitude_in_units, ureg, wraps_ufunc
from ..atomic_data import AtomicData
from .impurity_array_helpers import extend_impurity_concentration_array


@Algorithm.register_algorithm(
    return_keys=[
        "edge_impurity_concentration",
        "edge_impurity_concentration_in_core",
        "impurity_concentration",
    ]
)
def calc_edge_impurity_concentration(
    edge_impurity_species: AtomicSpecies,
    q_parallel: Unitfull,
    SOL_power_loss_fraction: Unitfull,
    target_electron_temp: Unitfull,
    separatrix_electron_temp: Unitfull,
    separatrix_electron_density: Unitfull,
    kappa_e0: Unitfull,
    lengyel_overestimation_factor: Unitfull,
    edge_impurity_enrichment: Unitfull,
    impurity_concentration: xr.DataArray,
    atomic_data: xr.DataArray | AtomicData,
    reference_electron_density: Unitfull = 1.0 * ureg.n20,
    reference_ne_tau: Unitfull = 1.0 * ureg.n20 * ureg.ms,
) -> tuple[Unitfull, ...]:
    """Calculate the impurity concentration required to cool the scrape-off-layer using the Lengyel model.

    Args:
        edge_impurity_species: :term:`glossary link<edge_impurity_species>`
        reference_electron_density: :term:`glossary link<reference_electron_density>`
        reference_ne_tau: :term:`glossary link<reference_ne_tau>`
        q_parallel: :term:`glossary link<q_parallel>`
        SOL_power_loss_fraction: :term:`glossary link<SOL_power_loss_fraction>`
        target_electron_temp: :term:`glossary link<target_electron_temp>`
        separatrix_electron_temp: :term:`glossary link<separatrix_electron_temp>`
        separatrix_electron_density: :term:`glossary link<separatrix_electron_density>`
        kappa_e0: :term:`glossary link<kappa_e0>`
        impurity_concentration: :term:`glossary link<impurity_concentration>`
        lengyel_overestimation_factor: :term:`glossary link<lengyel_overestimation_factor>`
        edge_impurity_enrichment: :term:`glossary link<edge_impurity_enrichment>`
        atomic_data: :term:`glossary link<atomic_data>`

    Returns:
        :term:`edge_impurity_concentration`, :term:`edge_impurity_concentration_in_core`, :term:`impurity_concentration`
    """
    L_int_integrator = build_L_int_integrator(
        atomic_data=get_item(atomic_data),
        impurity_species=edge_impurity_species,
        reference_electron_density=reference_electron_density,
        reference_ne_tau=reference_ne_tau,
    )

    edge_impurity_concentration = calc_edge_impurity_concentration_from_lengyel(
        L_int_integrator=L_int_integrator,
        q_parallel=q_parallel,
        SOL_power_loss_fraction=SOL_power_loss_fraction,
        target_electron_temp=target_electron_temp,
        separatrix_electron_temp=separatrix_electron_temp,
        separatrix_electron_density=separatrix_electron_density,
        kappa_e0=kappa_e0,
        lengyel_overestimation_factor=lengyel_overestimation_factor,
    )

    edge_impurity_concentration_in_core = edge_impurity_concentration / edge_impurity_enrichment
    impurity_concentration = extend_impurity_concentration_array(
        impurity_concentration, edge_impurity_species, edge_impurity_concentration_in_core
    )

    return (edge_impurity_concentration, edge_impurity_concentration_in_core, impurity_concentration)


def build_L_int_integrator(
    atomic_data: AtomicData,
    impurity_species: AtomicSpecies,
    reference_electron_density: Unitfull,
    reference_ne_tau: Unitfull,
) -> Callable[[Unitfull, Unitfull], Unitfull]:
    r"""Build an interpolator to calculate the integral of L_{int}$ between arbitrary temperature points.

    $L_int = \int_a^b L_z(T_e) sqrt(T_e) dT_e$ where $L_z$ is a cooling curve for an impurity species.
    This is used in the calculation of the radiated power associated with a given impurity.

    Args:
        atomic_data: :term:`glossary link<atomic_data>`
        impurity_species: [] :term:`glossary link<impurity_species>`
        reference_electron_density: [n20] :term:`glossary link<reference_electron_density>`
        reference_ne_tau: [n20 * ms] :term:`glossary link<reference_ne_tau>`

    Returns:
        L_int_integrator
    """
    Lz_curve = atomic_data.get_noncoronal_Lz_interpolator(get_item(impurity_species), ne_tau=reference_ne_tau)

    _electron_temp = Lz_curve.electron_temp_coord
    _electron_density = magnitude_in_units(reference_electron_density, Lz_curve.electron_density_units)
    assert np.size(_electron_density) == 1

    Lz_sqrt_Te = Lz_curve(electron_density=_electron_density, electron_temp=_electron_temp) * np.sqrt(_electron_temp)
    interpolator = InterpolatedUnivariateSpline(_electron_temp, magnitude(Lz_sqrt_Te))  # type: ignore[arg-type]

    def L_int(start_temp: float, stop_temp: float) -> float:
        integrated_Lz: float = interpolator.integral(start_temp, stop_temp)
        return integrated_Lz

    L_int_integrator: Callable[[Unitfull, Unitfull], Unitfull] = wraps_ufunc(
        input_units=dict(start_temp=ureg.eV, stop_temp=ureg.eV), return_units=dict(L_int=ureg.W * ureg.m**3 * ureg.eV**1.5)
    )(L_int)
    return L_int_integrator


def calc_edge_impurity_concentration_from_lengyel(
    L_int_integrator: Callable[[Unitfull, Unitfull], Unitfull],
    q_parallel: Unitfull,
    SOL_power_loss_fraction: Unitfull,
    target_electron_temp: Unitfull,
    separatrix_electron_temp: Unitfull,
    separatrix_electron_density: Unitfull,
    kappa_e0: Unitfull,
    lengyel_overestimation_factor: Unitfull,
) -> Unitfull:
    """Calculate the relative concentration of an edge impurity required to achieve a given SOL power loss fraction.

    N.b. this function does not ensure consistency of the calculated impurity concentration
    with the parallel temperature profile. You may wish to implement an iterative solver
    to find a consistent set of L_parallel, T_t and T_u.

    This model is based on the "Lengyel" model originally presented in a 1981 IPP report titled
    "Analysis of Radiating Plasma Boundary Layers" by L. L. Lengyel :cite:`Lengyel_1981`.

    The lengyel_overestimation_factor is introduced by D. Moulton et al. in :cite:`Moulton_2021`.
    This paper also provides a good description of the Lengyel model.

    Args:
        L_int_integrator: an interpolator to calculate the integral of L_{int}$ between arbitrary temperature points
        q_parallel: :term:`glossary link<q_parallel>`
        SOL_power_loss_fraction: :term:`glossary link<SOL_power_loss_fraction>`
        target_electron_temp: :term:`glossary link<target_electron_temp>`
        separatrix_electron_temp: :term:`glossary link<separatrix_electron_temp>`
        separatrix_electron_density: :term:`glossary link<separatrix_electron_density>`
        kappa_e0: :term:`glossary link<kappa_e0>`
        lengyel_overestimation_factor: :term:`glossary link<lengyel_overestimation_factor>`

    Returns:
        :term:`impurity_concentration`
    """
    L_int = L_int_integrator(target_electron_temp, separatrix_electron_temp)

    numerator = q_parallel**2 - ((1.0 - SOL_power_loss_fraction) * q_parallel) ** 2
    denominator = 2.0 * kappa_e0 * (separatrix_electron_density * separatrix_electron_temp) ** 2 * L_int

    return numerator / denominator / lengyel_overestimation_factor
