"""Calculate the thermal gain factor (Q, Q_plasma, Q_thermal)."""
import numpy as np

from ..algorithm_class import Algorithm
from ..unit_handling import Unitfull, ureg, wraps_ufunc

_IGNITED_THRESHOLD = 1e3
_IGNITED = 1e6


def _ignition_above_threshold(Q: float) -> float:
    """If Q > _IGNITED_THRESHOLD, set equal to _IGNITED.

    Args:
        Q: Fusion power gain [~]

    Returns:
         Q [~]
    """
    if Q > _IGNITED_THRESHOLD:
        return _IGNITED
    else:
        return Q


@Algorithm.register_algorithm(return_keys=["Q"])
@wraps_ufunc(return_units=dict(Q=ureg.dimensionless), input_units=dict(P_fusion=ureg.MW, P_launched=ureg.MW))
def thermal_calc_gain_factor(P_fusion: float, P_launched: float) -> float:
    """Calculate the fusion gain.

    Args:
        P_fusion: [MW] :term:`glossary link<P_fusion>`
        P_launched: [MW] :term:`glossary link<P_launched>`

    Returns:
         Q [~]
    """
    if np.isclose(P_launched, 0.0):
        return _IGNITED
    else:
        return _ignition_above_threshold(P_fusion / P_launched)


@Algorithm.register_algorithm(
    return_keys=[
        "P_external",
        "P_launched",
        "Q",
    ]
)
def run_calc_fusion_gain(
    P_in: Unitfull,
    P_alpha: Unitfull,
    P_fusion: Unitfull,
    fraction_of_external_power_coupled: Unitfull,
) -> dict[str, Unitfull]:
    """Calculate the fusion power and thermal gain (Q).

    Args:
        P_in: :term:`glossary link<P_in>`
        P_alpha: :term:`glossary link<P_alpha>`
        P_fusion: :term:`glossary link<P_fusion>`
        fraction_of_external_power_coupled: :term:`glossary link<fraction_of_external_power_coupled>`

    Returns:
        :term:`P_fusion`, :term:`P_neutron`, :term:`P_alpha`, :term:`P_external`, :term:`P_launched`, :term:`Q`, :term:`neutron_power_flux_to_walls` :term:`neutron_rate`
    """
    P_external = (P_in - P_alpha).clip(min=0.0 * ureg.MW)
    P_launched = P_external / fraction_of_external_power_coupled
    Q = thermal_calc_gain_factor(P_fusion, P_launched)

    return (
        P_external,
        P_launched,
        Q,
    )
