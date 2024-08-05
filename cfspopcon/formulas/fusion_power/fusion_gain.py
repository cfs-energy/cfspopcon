"""Calculate the fusion gain factor."""

from ...algorithm_class import Algorithm
from ...unit_handling import Unitfull, ureg


@Algorithm.register_algorithm(return_keys=["Q"])
def calc_fusion_gain(P_fusion: Unitfull, P_ohmic: Unitfull, P_auxillary_launched: Unitfull) -> Unitfull:
    """Calculate the fusion gain, using the launched power in the denominator.

    This is the thermal gain using the launched power.
    A slightly more optimistic Q can be obtained by using the absorbed power (P_external) in
    the denominator, but for scientific breakeven the launched power will be used.

    The denominator is forced to be at least 1W, to prevent a division-by-zero error.

    Args:
        P_fusion: [MW] :term:`glossary link<P_fusion>`
        P_ohmic: [MW] :term:`glossary link<P_ohmic>`
        P_auxillary_launched: [MW] :term:`glossary link<P_auxillary_launched>`

    Returns:
        :term:`Q` [~]
    """
    Q = P_fusion / (P_ohmic + P_auxillary_launched).clip(min=1.0 * ureg.W)

    return Q


@Algorithm.register_algorithm(return_keys=["fusion_triple_product"])
def calc_triple_product(peak_fuel_ion_density: Unitfull, peak_ion_temp: Unitfull, energy_confinement_time: Unitfull) -> Unitfull:
    """Calculate the fusion triple product.

    Args:
        peak_fuel_ion_density: [1e20 m^-3] :term:`glossary link<peak_fuel_ion_density>`
        peak_ion_temp: [keV] :term:`glossary link<peak_fuel_ion_density>`
        energy_confinement_time: [s] :term:`glossary link<energy_confinement_time>`

    Returns:
         fusion_triple_product [10e20 m**-3 keV s]
    """
    return peak_fuel_ion_density * peak_ion_temp * energy_confinement_time
