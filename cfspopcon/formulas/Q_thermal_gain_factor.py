"""Calculate the thermal gain factor (Q, Q_plasma, Q_thermal)."""
import numpy as np

from ..unit_handling import ureg, wraps_ufunc

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
