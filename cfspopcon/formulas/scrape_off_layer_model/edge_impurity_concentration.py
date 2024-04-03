"""Lengyel model to compute the edge impurity concentration."""
from typing import Callable

import numpy as np
import xarray as xr
from scipy.interpolate import InterpolatedUnivariateSpline  # type:ignore[import-untyped]

from ...algorithm_class import Algorithm
from ...named_options import AtomicSpecies
from ...read_atomic_data import AtomicData
from ...unit_handling import Unitfull, convert_units, magnitude, ureg, wraps_ufunc


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
    if isinstance(impurity_species, xr.DataArray):
        impurity_species = impurity_species.item()

    electron_density_ref = magnitude(convert_units(reference_electron_density, ureg.m**-3))
    ne_tau_ref = magnitude(convert_units(reference_ne_tau, ureg.m**-3 * ureg.s))

    Lz_curve = (
        atomic_data.get_dataset(impurity_species)
        .equilibrium_Lz.sel(dim_electron_density=electron_density_ref, method="nearest", tolerance=1e-6 * electron_density_ref)
        .sel(dim_ne_tau=ne_tau_ref, method="nearest", tolerance=1e-6 * ne_tau_ref)
    )

    electron_temp = Lz_curve.dim_electron_temp
    Lz_sqrt_Te = Lz_curve * np.sqrt(electron_temp)

    interpolator = InterpolatedUnivariateSpline(electron_temp, magnitude(Lz_sqrt_Te))

    def L_int(start_temp: float, stop_temp: float) -> float:
        return interpolator.integral(start_temp, stop_temp)  # type: ignore[no-any-return]

    return wraps_ufunc(input_units=dict(start_temp=ureg.eV, stop_temp=ureg.eV), return_units=dict(L_int=ureg.W * ureg.m**3 * ureg.eV**1.5))(L_int)  # type: ignore[no-any-return]


@Algorithm.register_algorithm(return_keys=["impurity_concentration"])
def calc_required_edge_impurity_concentration(
    L_int_integrator: Callable[[Unitfull, Unitfull], Unitfull],
    q_parallel: Unitfull,
    SOL_power_loss_fraction: Unitfull,
    target_electron_temp: Unitfull,
    upstream_electron_temp: Unitfull,
    upstream_electron_density: Unitfull,
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
        upstream_electron_temp: :term:`glossary link<upstream_electron_temp>`
        upstream_electron_density: :term:`glossary link<upstream_electron_density>`
        kappa_e0: :term:`glossary link<kappa_e0>`
        lengyel_overestimation_factor: :term:`glossary link<lengyel_overestimation_factor>`

    Returns:
        :term:`impurity_concentration`
    """
    L_int = L_int_integrator(target_electron_temp, upstream_electron_temp)

    numerator = q_parallel**2 - ((1.0 - SOL_power_loss_fraction) * q_parallel) ** 2
    denominator = 2.0 * kappa_e0 * (upstream_electron_density * upstream_electron_temp) ** 2 * L_int

    return numerator / denominator / lengyel_overestimation_factor
