"""Calculate the fusion gain factor."""

from ...algorithm_class import Algorithm
from ...unit_handling import Unitfull, ureg


@Algorithm.register_algorithm(return_keys=["P_external", "P_launched", "Q"])
def calc_fusion_gain(
    P_fusion: Unitfull,
    P_alpha: Unitfull,
    P_in: Unitfull,
    fraction_of_external_power_coupled: Unitfull,
) -> Unitfull:
    """Calculate the fusion gain, using the launched power in the denominator.

    This is the thermal gain using the launched power.
    A slightly more optimistic Q can be obtained by using the absorbed power (P_external) in
    the denominator, but for scientific breakeven the launched power will be used.

    Args:
        P_fusion: [MW] :term:`glossary link<P_fusion>`
        P_alpha: [MW] :term:`glossary link<P_alpha>`
        P_in: :term:`glossary link<P_in>`
        fraction_of_external_power_coupled: :term:`glossary link<fraction_of_external_power_coupled>`

    Returns:
        :term:`P_external` [MW], :term:`P_launched` [MW], :term:`Q` [~]
    """
    P_external = (P_in - P_alpha).clip(min=0.0 * ureg.MW)
    P_launched = P_external / fraction_of_external_power_coupled

    Q = P_fusion / P_launched

    # #TODO: remove this, it's weird. Just set a max Q.
    _IGNITED = 1e6
    _IGNITED_THRESHOLD = 1e3
    Q = Q.where(Q < _IGNITED_THRESHOLD, _IGNITED)

    return P_external, P_launched, Q


@Algorithm.register_algorithm(return_keys=["fusion_triple_product"])
def calc_triple_product(
    peak_fuel_ion_density: Unitfull,
    peak_ion_temp: Unitfull,
    energy_confinement_time: Unitfull,
) -> Unitfull:
    """Calculate the fusion triple product.

    Args:
        peak_fuel_ion_density: [1e20 m^-3] :term:`glossary link<peak_fuel_ion_density>`
        peak_ion_temp: [keV] :term:`glossary link<peak_fuel_ion_density>`
        energy_confinement_time: [s] :term:`glossary link<energy_confinement_time>`

    Returns:
         fusion_triple_product [10e20 m**-3 keV s]
    """
    return peak_fuel_ion_density * peak_ion_temp * energy_confinement_time
