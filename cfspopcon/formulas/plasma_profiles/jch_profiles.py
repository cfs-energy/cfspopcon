from bisect import bisect_left
from collections.abc import Callable
from typing import cast

import numpy as np
import xarray as xr
from scipy.optimize import brentq

from ...algorithm_class import Algorithm
from ...unit_handling import ureg, magnitude_in_units


@Algorithm.register_algorithm(return_keys=["rho", "electron_density_profile", "fuel_ion_density_profile", "electron_temp_profile", "ion_temp_profile"])
def calc_jch_profiles(
    average_electron_density: float,
    average_electron_temp: float,
    average_ion_temp: float,
    electron_density_peaking: float,
    ion_density_peaking: float,
    temperature_peaking: float,
    dilution: float,
    n_points_for_confined_region_profiles: int = 50,
    pedestal_width: float = 0.05 * ureg.dimensionless,
    t_sep: float = 0.2 * ureg.keV,
    n_sep_ratio: float = 0.5 * ureg.dimensionless,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate JCH profiles with an exponential core and linear pedestal handoff.

    The public peaking inputs remain center-to-volume-average ratios. This
    helper converts them to the peak-to-pedestal ratios required by the JCH
    profile parameterization. Density and temperature branches can be requested
    independently; the unused branch returns ``None`` so the caller can splice
    together mixed-form runs.
    """
    n_points = int(n_points_for_confined_region_profiles)
    pedestal_width = magnitude_in_units(pedestal_width, ureg.dimensionless)
    separatrix_temperature = magnitude_in_units(t_sep, ureg.keV)
    separatrix_to_pedestal_ratio = magnitude_in_units(n_sep_ratio, ureg.dimensionless)

    if n_points < 3:
        raise ValueError("JCH profiles require at least three radial grid points.")
    if not 0.0 < pedestal_width < 1.0:
        raise ValueError("pedestal_width must lie strictly between 0 and 1 for JCH profiles.")

    electron_temp_peaking = ion_temp_peaking = temperature_peaking

    rho_ped = 1.0 - pedestal_width
    rho = _build_profile_grid(n_points, rho_ped)
    rho = xr.DataArray(rho, coords=dict(dim_rho=rho))

    rho_core = rho[rho <= rho_ped]
    edge_basis_1, edge_basis_2, edge_integral_1, edge_integral_2 = _calc_jch_edge_integrals(rho, rho_ped, pedestal_width)

    electron_density_pedestal_peaking = _solve_jch_density_pedestal_peaking(
        target_volume_peaking=magnitude_in_units(electron_density_peaking, ureg.dimensionless),
        rho_core=rho_core,
        rho_ped=rho_ped,
        edge_integral_1=edge_integral_1,
        edge_integral_2=edge_integral_2,
        separatrix_to_pedestal_ratio=separatrix_to_pedestal_ratio,
    )
    ion_density_pedestal_peaking = _solve_jch_density_pedestal_peaking(
        target_volume_peaking=magnitude_in_units(ion_density_peaking, ureg.dimensionless),
        rho_core=rho_core,
        rho_ped=rho_ped,
        edge_integral_1=edge_integral_1,
        edge_integral_2=edge_integral_2,
        separatrix_to_pedestal_ratio=separatrix_to_pedestal_ratio,
    )
    electron_density_profile = _build_jch_density_profile(
        volume_average=magnitude_in_units(average_electron_density, ureg.n19),
        peak_to_pedestal=electron_density_pedestal_peaking,
        rho=rho,
        rho_ped=rho_ped,
        edge_basis_1=edge_basis_1,
        edge_basis_2=edge_basis_2,
        edge_integral_1=edge_integral_1,
        edge_integral_2=edge_integral_2,
        separatrix_to_pedestal_ratio=separatrix_to_pedestal_ratio,
    ) * ureg.n19
    fuel_ion_density_profile = _build_jch_density_profile(
        volume_average=magnitude_in_units(average_electron_density * dilution, ureg.n19),
        peak_to_pedestal=ion_density_pedestal_peaking,
        rho=rho,
        rho_ped=rho_ped,
        edge_basis_1=edge_basis_1,
        edge_basis_2=edge_basis_2,
        edge_integral_1=edge_integral_1,
        edge_integral_2=edge_integral_2,
        separatrix_to_pedestal_ratio=separatrix_to_pedestal_ratio,
    ) * ureg.n19

    electron_temp_pedestal_peaking = _solve_jch_temperature_pedestal_peaking(
        target_volume_peaking=magnitude_in_units(electron_temp_peaking, ureg.dimensionless),
        volume_average=magnitude_in_units(average_electron_temp, ureg.keV),
        rho_core=rho_core,
        rho_ped=rho_ped,
        edge_integral_1=edge_integral_1,
        edge_integral_2=edge_integral_2,
        separatrix_temperature=separatrix_temperature,
    )
    ion_temp_pedestal_peaking = _solve_jch_temperature_pedestal_peaking(
        target_volume_peaking=magnitude_in_units(ion_temp_peaking, ureg.dimensionless),
        volume_average=magnitude_in_units(average_ion_temp, ureg.keV),
        rho_core=rho_core,
        rho_ped=rho_ped,
        edge_integral_1=edge_integral_1,
        edge_integral_2=edge_integral_2,
        separatrix_temperature=separatrix_temperature,
    )
    electron_temp_profile = _build_jch_temperature_profile(
        volume_average=magnitude_in_units(average_electron_temp, ureg.keV),
        peak_to_pedestal=electron_temp_pedestal_peaking,
        rho=rho,
        rho_ped=rho_ped,
        edge_basis_1=edge_basis_1,
        edge_basis_2=edge_basis_2,
        edge_integral_1=edge_integral_1,
        edge_integral_2=edge_integral_2,
        separatrix_temperature=separatrix_temperature,
    ) * ureg.keV
    ion_temp_profile = _build_jch_temperature_profile(
        volume_average=magnitude_in_units(average_ion_temp, ureg.keV),
        peak_to_pedestal=ion_temp_pedestal_peaking,
        rho=rho,
        rho_ped=rho_ped,
        edge_basis_1=edge_basis_1,
        edge_basis_2=edge_basis_2,
        edge_integral_1=edge_integral_1,
        edge_integral_2=edge_integral_2,
        separatrix_temperature=separatrix_temperature,
    ) * ureg.keV

    return rho, electron_density_profile, fuel_ion_density_profile, electron_temp_profile, ion_temp_profile

@Algorithm.register_algorithm(return_keys=["electron_density_pedestal_peaking", "ion_density_pedestal_peaking", "electron_temp_pedestal_peaking", "ion_temp_pedestal_peaking"])
def calc_jch_pedestal_peaking(
    average_electron_temp: float,
    average_ion_temp: float,
    electron_density_peaking: float,
    ion_density_peaking: float,
    temperature_peaking: float,
    n_points_for_confined_region_profiles: int = 50,
    pedestal_width: float = 0.05 * ureg.dimensionless,
    t_sep: float = 0.2 * ureg.keV,
    n_sep_ratio: float = 0.5 * ureg.dimensionless,
) -> tuple[float, float, float, float]:
    """Convert volume peaking values into the JCH peak-to-pedestal ratios.

    The public API reports peaking as center-to-volume-average ratios for every
    profile family. JCH profile construction instead needs center-to-pedestal
    ratios, so this helper performs the inversion using the same pedestal
    geometry and integration rules as :func:`calc_jch_profiles`.

    """
    pedestal_width = float(pedestal_width)
    rho_ped = 1.0 - pedestal_width
    rho = _build_profile_grid(int(n_points_for_confined_region_profiles), rho_ped)
    rho_core = rho[rho <= rho_ped]
    _, _, edge_integral_1, edge_integral_2 = _calc_jch_edge_integrals(rho, rho_ped, pedestal_width)

    electron_density_pedestal_peaking = _solve_jch_density_pedestal_peaking(
        target_volume_peaking=magnitude_in_units(electron_density_peaking, ureg.dimensionless),
        rho_core=rho_core,
        rho_ped=rho_ped,
        edge_integral_1=edge_integral_1,
        edge_integral_2=edge_integral_2,
        separatrix_to_pedestal_ratio=magnitude_in_units(n_sep_ratio, ureg.dimensionless),
    )
    ion_density_pedestal_peaking = _solve_jch_density_pedestal_peaking(
        target_volume_peaking=magnitude_in_units(ion_density_peaking, ureg.dimensionless),
        rho_core=rho_core,
        rho_ped=rho_ped,
        edge_integral_1=edge_integral_1,
        edge_integral_2=edge_integral_2,
        separatrix_to_pedestal_ratio=magnitude_in_units(n_sep_ratio, ureg.dimensionless),
    )
    electron_temp_pedestal_peaking = _solve_jch_temperature_pedestal_peaking(
        target_volume_peaking=magnitude_in_units(temperature_peaking, ureg.dimensionless),
        volume_average=magnitude_in_units(average_electron_temp, ureg.keV),
        rho_core=rho_core,
        rho_ped=rho_ped,
        edge_integral_1=edge_integral_1,
        edge_integral_2=edge_integral_2,
        separatrix_temperature=magnitude_in_units(t_sep, ureg.keV),
    )
    ion_temp_pedestal_peaking = _solve_jch_temperature_pedestal_peaking(
        target_volume_peaking=magnitude_in_units(temperature_peaking, ureg.dimensionless),
        volume_average=magnitude_in_units(average_ion_temp, ureg.keV),
        rho_core=rho_core,
        rho_ped=rho_ped,
        edge_integral_1=edge_integral_1,
        edge_integral_2=edge_integral_2,
        separatrix_temperature=magnitude_in_units(t_sep, ureg.keV),
    )

    return (
        electron_density_pedestal_peaking,
        ion_density_pedestal_peaking,
        electron_temp_pedestal_peaking,
        ion_temp_pedestal_peaking,
    )

def _calc_jch_core_integral(gradient: float, rho_core: np.ndarray, rho_ped: float) -> float:
    """Integrate a pedestal-normalized exponential core profile over the confined volume."""
    profile = np.exp(gradient * (rho_ped - rho_core))
    return float(np.trapezoid(profile * 2.0 * rho_core, x=rho_core))

def _calc_jch_edge_integrals(rho: np.ndarray, rho_ped: float, pedestal_width: float) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Compute the linear pedestal basis functions and their volume integrals.

    The JCH edge is represented as a linear blend between the pedestal value and
    the separatrix anchor. Returning both basis arrays and their integrals lets
    the solver and profile builder share the same geometry bookkeeping.
    """
    rho_edge = rho[rho >= rho_ped]
    edge_basis_1 = (1.0 - rho_edge) / pedestal_width
    edge_basis_2 = (rho_edge - rho_ped) / pedestal_width

    edge_integral_1 = float(np.trapezoid(edge_basis_1 * 2.0 * rho_edge, x=rho_edge))
    edge_integral_2 = float(np.trapezoid(edge_basis_2 * 2.0 * rho_edge, x=rho_edge))

    return edge_basis_1, edge_basis_2, edge_integral_1, edge_integral_2

def _calc_jch_density_volume_peaking(
    peak_to_pedestal: float,
    rho_core: np.ndarray,
    rho_ped: float,
    edge_integral_1: float,
    edge_integral_2: float,
    separatrix_to_pedestal_ratio: float,
) -> float:
    """Return the peak-to-volume-average ratio for a JCH density profile.

    ``peak_to_pedestal`` is the natural JCH input. This helper translates it
    into the public volume-peaking definition used elsewhere in the codebase.
    """
    gradient = float(np.log(peak_to_pedestal) / rho_ped)
    core_integral = _calc_jch_core_integral(gradient, rho_core, rho_ped)
    return peak_to_pedestal / (core_integral + edge_integral_1 + separatrix_to_pedestal_ratio * edge_integral_2)

def _calc_jch_temperature_pedestal_temperature(
    volume_average: float,
    peak_to_pedestal: float,
    rho_core: np.ndarray,
    rho_ped: float,
    edge_integral_1: float,
    edge_integral_2: float,
    separatrix_temperature: float,
) -> float:
    """Return the pedestal temperature implied by a JCH peak-to-pedestal ratio.

    Temperature profiles are additionally constrained by the separatrix
    temperature, so the pedestal value must be solved from the requested volume
    average before the profile can be constructed.
    """
    gradient = float(np.log(peak_to_pedestal) / rho_ped)
    core_integral = _calc_jch_core_integral(gradient, rho_core, rho_ped)
    return (volume_average - separatrix_temperature * edge_integral_2) / (core_integral + edge_integral_1)

def _calc_jch_temperature_volume_peaking(
    volume_average: float,
    peak_to_pedestal: float,
    rho_core: np.ndarray,
    rho_ped: float,
    edge_integral_1: float,
    edge_integral_2: float,
    separatrix_temperature: float,
) -> float:
    """Return the peak-to-volume-average ratio for a JCH temperature profile."""
    pedestal_temperature = _calc_jch_temperature_pedestal_temperature(
        volume_average=volume_average,
        peak_to_pedestal=peak_to_pedestal,
        rho_core=rho_core,
        rho_ped=rho_ped,
        edge_integral_1=edge_integral_1,
        edge_integral_2=edge_integral_2,
        separatrix_temperature=separatrix_temperature,
    )
    return peak_to_pedestal * pedestal_temperature / volume_average

def _solve_jch_peak_to_pedestal(
    target_volume_peaking: float,
    volume_peaking_function: Callable[[float], float],
    quantity_name: str,
    maximum_peak_to_pedestal: float | None = None,
) -> float:
    """Solve for the JCH peak-to-pedestal ratio that matches a target volume peaking.

    The mapping from peak-to-pedestal to peak-to-volume is monotonic but does
    not have a closed-form inverse here, so we bracket the valid range and use a
    scalar root find. Temperature profiles can optionally impose an additional
    upper bound from the separatrix-temperature constraint.
    """
    minimum_volume_peaking = volume_peaking_function(1.0)
    if target_volume_peaking < minimum_volume_peaking and not np.isclose(target_volume_peaking, minimum_volume_peaking):
        raise ValueError(
            f"Requested JCH {quantity_name} peaking {target_volume_peaking} is below the minimum compatible "
            f"volume peaking {minimum_volume_peaking}."
        )
    if np.isclose(target_volume_peaking, minimum_volume_peaking):
        return 1.0

    if maximum_peak_to_pedestal is None:
        lower_bound = 1.0
        upper_bound = 2.0
        while volume_peaking_function(upper_bound) < target_volume_peaking:
            lower_bound, upper_bound = upper_bound, upper_bound * 2.0
            if upper_bound > 1.0e12:
                raise ValueError(f"Could not bracket the requested JCH {quantity_name} peaking {target_volume_peaking}.")
    else:
        lower_bound = 1.0
        upper_bound = maximum_peak_to_pedestal
        maximum_volume_peaking = volume_peaking_function(upper_bound)
        if target_volume_peaking > maximum_volume_peaking and not np.isclose(target_volume_peaking, maximum_volume_peaking):
            raise ValueError(
                f"Requested JCH {quantity_name} peaking {target_volume_peaking} exceeds the maximum compatible "
                f"volume peaking {maximum_volume_peaking}."
            )
        if np.isclose(target_volume_peaking, maximum_volume_peaking):
            return upper_bound

    return float(
        brentq(lambda peak_to_pedestal: volume_peaking_function(peak_to_pedestal) - target_volume_peaking, lower_bound, upper_bound)
    )

def _solve_jch_density_pedestal_peaking(
    target_volume_peaking: float,
    rho_core: np.ndarray,
    rho_ped: float,
    edge_integral_1: float,
    edge_integral_2: float,
    separatrix_to_pedestal_ratio: float,
) -> float:
    """Convert a density peak-to-average ratio into the JCH peak-to-pedestal ratio."""
    return _solve_jch_peak_to_pedestal(
        target_volume_peaking=target_volume_peaking,
        volume_peaking_function=lambda peak_to_pedestal: _calc_jch_density_volume_peaking(
            peak_to_pedestal=peak_to_pedestal,
            rho_core=rho_core,
            rho_ped=rho_ped,
            edge_integral_1=edge_integral_1,
            edge_integral_2=edge_integral_2,
            separatrix_to_pedestal_ratio=separatrix_to_pedestal_ratio,
        ),
        quantity_name="density",
    )

def _solve_jch_temperature_pedestal_peaking(
    target_volume_peaking: float,
    volume_average: float,
    rho_core: np.ndarray,
    rho_ped: float,
    edge_integral_1: float,
    edge_integral_2: float,
    separatrix_temperature: float,
) -> float:
    """Convert a temperature peak-to-average ratio into the JCH peak-to-pedestal ratio."""

    def volume_peaking_function(peak_to_pedestal: float) -> float:
        return _calc_jch_temperature_volume_peaking(
            volume_average=volume_average,
            peak_to_pedestal=peak_to_pedestal,
            rho_core=rho_core,
            rho_ped=rho_ped,
            edge_integral_1=edge_integral_1,
            edge_integral_2=edge_integral_2,
            separatrix_temperature=separatrix_temperature,
        )

    maximum_peak_to_pedestal: float | None = None
    if separatrix_temperature > 0.0:
        # First ensure the flat-core limit is already physical. If not, no valid
        # peak-to-pedestal ratio exists for the requested average temperature.
        if (
            _calc_jch_temperature_pedestal_temperature(
                volume_average=volume_average,
                peak_to_pedestal=1.0,
                rho_core=rho_core,
                rho_ped=rho_ped,
                edge_integral_1=edge_integral_1,
                edge_integral_2=edge_integral_2,
                separatrix_temperature=separatrix_temperature,
            )
            < separatrix_temperature
        ):
            raise ValueError("Requested JCH temperature profile gives an unphysical pedestal below the separatrix temperature.")

        lower_bound = 1.0
        upper_bound = 2.0
        # Find the largest peak-to-pedestal ratio that still leaves the
        # pedestal at or above the separatrix temperature.
        while (
            _calc_jch_temperature_pedestal_temperature(
                volume_average=volume_average,
                peak_to_pedestal=upper_bound,
                rho_core=rho_core,
                rho_ped=rho_ped,
                edge_integral_1=edge_integral_1,
                edge_integral_2=edge_integral_2,
                separatrix_temperature=separatrix_temperature,
            )
            >= separatrix_temperature
        ):
            lower_bound, upper_bound = upper_bound, upper_bound * 2.0
            if upper_bound > 1.0e12:
                raise ValueError("Could not bracket the maximum valid JCH temperature pedestal peaking.")

        maximum_peak_to_pedestal = float(
            brentq(
                lambda peak_to_pedestal: _calc_jch_temperature_pedestal_temperature(
                    volume_average=volume_average,
                    peak_to_pedestal=peak_to_pedestal,
                    rho_core=rho_core,
                    rho_ped=rho_ped,
                    edge_integral_1=edge_integral_1,
                    edge_integral_2=edge_integral_2,
                    separatrix_temperature=separatrix_temperature,
                )
                - separatrix_temperature,
                lower_bound,
                upper_bound,
            )
        )

    return _solve_jch_peak_to_pedestal(
        target_volume_peaking=target_volume_peaking,
        volume_peaking_function=volume_peaking_function,
        quantity_name="temperature",
        maximum_peak_to_pedestal=maximum_peak_to_pedestal,
    )

def _build_jch_density_profile(
    volume_average: float,
    peak_to_pedestal: float,
    rho: np.ndarray,
    rho_ped: float,
    edge_basis_1: np.ndarray,
    edge_basis_2: np.ndarray,
    edge_integral_1: float,
    edge_integral_2: float,
    separatrix_to_pedestal_ratio: float,
) -> np.ndarray:
    """Construct a density profile with a requested center-to-pedestal ratio.

    The pedestal value is solved so that the full piecewise profile integrates
    back to the requested volume average.
    """
    rho_core = rho[rho <= rho_ped]
    rho_edge = rho[rho >= rho_ped]
    gradient = float(np.log(peak_to_pedestal) / rho_ped)
    core_integral = _calc_jch_core_integral(gradient, rho_core, rho_ped)
    pedestal_value = volume_average / (core_integral + edge_integral_1 + separatrix_to_pedestal_ratio * edge_integral_2)

    profile = np.empty_like(rho)
    profile[rho <= rho_ped] = pedestal_value * np.exp(gradient * (rho_ped - rho_core))
    if rho_edge.size:
        profile[rho >= rho_ped] = pedestal_value * edge_basis_1 + (pedestal_value * separatrix_to_pedestal_ratio) * edge_basis_2

    return profile

def _build_jch_temperature_profile(
    volume_average: float,
    peak_to_pedestal: float,
    rho: np.ndarray,
    rho_ped: float,
    edge_basis_1: np.ndarray,
    edge_basis_2: np.ndarray,
    edge_integral_1: float,
    edge_integral_2: float,
    separatrix_temperature: float,
) -> np.ndarray:
    """Construct a temperature profile with a requested center-to-pedestal ratio.

    Unlike density, the edge branch is anchored to an absolute separatrix
    temperature, so the pedestal temperature must be solved before the profile
    can be filled in.
    """
    rho_core = rho[rho <= rho_ped]
    rho_edge = rho[rho >= rho_ped]
    gradient = float(np.log(peak_to_pedestal) / rho_ped)
    pedestal_temperature = _calc_jch_temperature_pedestal_temperature(
        volume_average=volume_average,
        peak_to_pedestal=peak_to_pedestal,
        rho_core=rho_core,
        rho_ped=rho_ped,
        edge_integral_1=edge_integral_1,
        edge_integral_2=edge_integral_2,
        separatrix_temperature=separatrix_temperature,
    )

    if pedestal_temperature < separatrix_temperature:
        raise ValueError("Requested JCH temperature profile gives an unphysical pedestal below the separatrix temperature.")

    profile = np.empty_like(rho)
    profile[rho <= rho_ped] = pedestal_temperature * np.exp(gradient * (rho_ped - rho_core))
    if rho_edge.size:
        profile[rho >= rho_ped] = pedestal_temperature * edge_basis_1 + separatrix_temperature * edge_basis_2

    return profile

def _find_nearest_grid_index(values: np.ndarray, target: float) -> int:
    """Find the nearest point in a sorted grid using a bisection search.

    This is used when an externally supplied grid needs to be snapped to the
    exact JCH pedestal knee without scanning the whole array.
    """
    insertion_index = bisect_left(values.tolist(), target)

    if insertion_index == 0:
        return 0
    if insertion_index == len(values):
        return len(values) - 1
    if (values[insertion_index] - target) < (target - values[insertion_index - 1]):
        return insertion_index

    return insertion_index - 1

def _find_nearest_interior_grid_index(values: np.ndarray, target: float) -> int:
    """Find the nearest interior point in a sorted grid.

    JCH profile construction reserves the axis and edge points, so the pedestal
    knee must be snapped onto an interior point only.
    """
    if len(values) < 3:
        raise ValueError("JCH profiles require at least three radial grid points to preserve both the axis and separatrix.")

    return 1 + _find_nearest_grid_index(values[1:-1], target)

def _calc_profile_grid_edge_nudge(npoints: int) -> float:
    """Return the resolution-dependent LCFS offset used to regularize the grid.

    Hollow analytic profiles are singular exactly at ``rho = 1``. Rather than
    using one fixed epsilon, the nudge is scaled to the grid spacing so the last
    trapezoid stays well behaved across different resolutions.
    """
    if npoints <= 1:
        return 0.0

    # Choose the endpoint offset so it is one tenth of the induced grid
    # spacing: nudge = 0.1 * drho, drho = (1 - nudge) / (npoints - 1).
    return 0.1 / (npoints - 1 + 0.1)

def _build_profile_grid(npoints: int, rho_ped: float | None = None) -> np.ndarray:
    """Build the radial grid and optionally reserve four points across the pedestal.

    Non-JCH grids stop about one tenth of a grid spacing inside the LCFS so the
    analytic hollow-profile form is regularized without overweighting the final
    trapezoid. JCH grids use the same offset so mixed analytic/JCH calls can
    safely share a single grid without evaluating analytic hollow profiles at
    ``rho = 1``.
    """
    edge_nudge = _calc_profile_grid_edge_nudge(npoints)

    if rho_ped is None:
        return np.linspace(0.0, 1.0 - edge_nudge, num=npoints)

    # Reserve four samples for the pedestal region: the knee, two interior
    # points that can capture some curvature, and the edge point.
    pedestal_points = 4
    if npoints < pedestal_points + 1:
        raise ValueError("JCH profile grids require at least five radial points to preserve the axis and four pedestal samples.")

    core_points = npoints - pedestal_points + 1
    rho_core = np.linspace(0.0, rho_ped, num=core_points)
    rho_pedestal = np.linspace(rho_ped, 1.0 - edge_nudge, num=pedestal_points)
    return cast("FloatArray", np.concatenate((rho_core, rho_pedestal[1:])))


# def _remap_profile_onto_grid(
#     profile: np.ndarray,
#     source_rho: np.ndarray,
#     target_rho: np.ndarray,
#     target_volume_average: float,
# ) -> np.ndarray:
#     """Interpolate a profile onto a new rho grid and renormalize its volume average.

#     Mixed-form runs keep one public ``rho`` output even when different profile
#     families need different construction grids internally. This helper is what
#     lets PRF keep its own native construction grid and then move onto the common
#     output grid without losing the requested volume average.
#     """
#     if np.allclose(source_rho, target_rho):
#         return profile

#     remapped_profile = cast("FloatArray", np.interp(target_rho, source_rho, profile))
#     if np.isclose(target_volume_average, 0.0):
#         return np.zeros_like(remapped_profile)

#     # Renormalize with the same cylindrical-volume measure used elsewhere in
#     # the profile code so the remapped profile still hits the requested average.
#     remapped_volume_average = float(np.trapezoid(remapped_profile * 2.0 * target_rho, x=target_rho))
#     if np.isclose(remapped_volume_average, 0.0):
#         raise ValueError("Cannot renormalize a remapped profile with zero volume average.")

#     return remapped_profile * (target_volume_average / remapped_volume_average)
