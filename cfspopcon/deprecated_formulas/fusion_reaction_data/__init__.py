"""Reactions rates and power densities for various fusion reactions."""
from typing import Callable, Union

from numpy import float64
from numpy.typing import NDArray

from ...named_options import ReactionType
from .reaction_energies import reaction_energy_DD, reaction_energy_DHe3, reaction_energy_DT, reaction_energy_pB11
from .reaction_rate_coefficients import sigmav_DD, sigmav_DD_BoschHale, sigmav_DHe3, sigmav_DT, sigmav_DT_BoschHale, sigmav_pB11

SIGMAV: dict[
    ReactionType,
    Union[
        Callable[[NDArray[float64]], NDArray[float64]],
        Callable[[NDArray[float64]], tuple[NDArray[float64], NDArray[float64], NDArray[float64]]],
    ],
] = {
    ReactionType.DT: sigmav_DT,
    ReactionType.DD: sigmav_DD_BoschHale,
    ReactionType.DHe3: sigmav_DHe3,
    ReactionType.pB11: sigmav_pB11,
}

ENERGY: dict[
    ReactionType,
    Union[
        Callable[[NDArray[float64], float], tuple[float, float, float, NDArray[float64], NDArray[float64], NDArray[float64]]],
        Callable[
            [tuple[NDArray[float64], NDArray[float64], NDArray[float64]]],
            tuple[NDArray[float64], float, NDArray[float64], NDArray[float64], NDArray[float64], NDArray[float64]],
        ],
    ],
] = {
    ReactionType.DT: reaction_energy_DT,
    ReactionType.DD: reaction_energy_DD,
    ReactionType.DHe3: reaction_energy_DHe3,
    ReactionType.pB11: reaction_energy_pB11,
}

__all__ = [
    "SIGMAV",
    "ENERGY",
    "reaction_energy_DD",
    "reaction_energy_DHe3",
    "reaction_energy_DT",
    "reaction_energy_pB11",
    "sigmav_DD_BoschHale",
    "sigmav_DHe3",
    "sigmav_DT",
    "sigmav_pB11",
    "sigmav_DD",
    "sigmav_DT_BoschHale",
]
