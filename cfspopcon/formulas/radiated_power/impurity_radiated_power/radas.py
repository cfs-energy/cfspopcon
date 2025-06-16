"""Calculate the impurity radiated power using the radas atomic_data."""

from ....algorithm_class import Algorithm
from ....helpers import get_item
from ....unit_handling import Unitfull, ureg
from ...atomic_data import AtomicData
from ...geometry.volume_integral import integrate_profile_over_volume


@Algorithm.register_algorithm(return_keys=["P_rad_impurity"])
def calc_impurity_radiated_power_radas(
    rho: Unitfull,
    electron_temp_profile: Unitfull,
    electron_density_profile: Unitfull,
    impurity_concentration: Unitfull,
    plasma_volume: Unitfull,
    atomic_data: AtomicData,
) -> Unitfull:
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

    def calc_radiated_power_for_one_species(impurity_concentration: Unitfull) -> Unitfull:
        interpolator = atomic_data.get_coronal_Lz_interpolator(get_item(impurity_concentration.dim_species))

        Lz = interpolator.vector_eval(electron_density=electron_density_profile, electron_temp=electron_temp_profile, allow_extrap=True)
        radiated_power_profile = impurity_concentration * electron_density_profile**2 * Lz

        return integrate_profile_over_volume(radiated_power_profile / ureg.MW, rho, plasma_volume) * ureg.MW

    return impurity_concentration.groupby("dim_species").map(calc_radiated_power_for_one_species)
