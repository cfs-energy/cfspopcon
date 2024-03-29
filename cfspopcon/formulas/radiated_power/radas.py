"""Calculate the impurity radiated power using the radas atomic_data."""
from numpy import float64
from numpy.typing import NDArray

from ...named_options import AtomicSpecies
from ...read_atomic_data import AtomicData
from ...unit_handling import ureg, wraps_ufunc
from ..helpers import integrate_profile_over_volume


@wraps_ufunc(
    return_units=dict(radiated_power=ureg.MW),
    input_units=dict(
        rho=ureg.dimensionless,
        electron_temp_profile=ureg.eV,
        electron_density_profile=ureg.m**-3,
        impurity_concentration=ureg.dimensionless,
        impurity_species=None,
        plasma_volume=ureg.m**3,
        atomic_data=None,
    ),
    input_core_dims=[("dim_rho",), ("dim_rho",), ("dim_rho",), (), (), ()],
    pass_as_kwargs=("atomic_data",),
)
def calc_impurity_radiated_power_radas(
    rho: NDArray[float64],
    electron_temp_profile: NDArray[float64],
    electron_density_profile: NDArray[float64],
    impurity_concentration: float,
    impurity_species: AtomicSpecies,
    plasma_volume: float,
    atomic_data: AtomicData,
) -> float:
    """Calculation of radiated power using radas atomic_data datasets.

    Args:
        rho: [~] :term:`glossary link<rho>`
        electron_temp_profile: [eV] :term:`glossary link<electron_temp_profile>`
        electron_density_profile: [m^-3] :term:`glossary link<electron_density_profile>`
        impurity_species: [] :term:`glossary link<impurity_species>`
        impurity_concentration: [~] :term:`glossary link<impurity_concentration>`
        plasma_volume: [m^3] :term:`glossary link<plasma_volume>`
        atomic_data: :term:`glossary link<atomic_data>`

    Returns:
         [MW] Estimated radiation power due to this impurity
    """
    MW_per_W = 1e6

    Lz = atomic_data.eval_interpolator(
        electron_density=electron_density_profile,
        electron_temp=electron_temp_profile,
        kind=AtomicData.CoronalLz,
        species=impurity_species,
        grid=False,
        allow_extrapolation=True,
        coords=dict(dim_rho=rho),
    )
    radiated_power_profile = electron_density_profile**2 * Lz

    radiated_power = impurity_concentration * integrate_profile_over_volume(radiated_power_profile, rho, plasma_volume) / MW_per_W

    return radiated_power
