"""Calculate simple metrics for the heat exhaust challenge."""

from ...algorithm_class import Algorithm
from ...unit_handling import Unitfull


@Algorithm.register_algorithm(return_keys=["PB_over_R"])
def calc_PB_over_R(
    P_sol: Unitfull, magnetic_field_on_axis: Unitfull, major_radius: Unitfull
) -> Unitfull:
    """Calculate P_sep*B0/R0, which scales roughly the same as the parallel heat flux density entering the scrape-off-layer."""
    return P_sol * magnetic_field_on_axis / major_radius


@Algorithm.register_algorithm(return_keys=["PBpRnSq"])
def calc_PBpRnSq(
    P_sol: Unitfull,
    magnetic_field_on_axis: Unitfull,
    q_star: Unitfull,
    major_radius: Unitfull,
    average_electron_density: Unitfull,
) -> Unitfull:
    """Calculate P_sep * B_pol / (R * n^2), which scales roughly the same as the impurity fraction required for detachment."""
    return (P_sol * (magnetic_field_on_axis / q_star) / major_radius) / (
        average_electron_density**2.0
    )
