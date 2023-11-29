"""Routines to calculate the separatrix power required to reach the LH transition."""
import matplotlib.pyplot as plt  # type:ignore[import]
import numpy as np
import xarray as xr
from scipy.interpolate import RectBivariateSpline  # type:ignore[import]

from cfspopcon.unit_handling import Unitfull, convert_units, ureg


def extract_LH_contour_points(LH_transition_condition: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """Extract the density and temperature along the LH transition line."""
    # Hacky way of getting contour points
    _, ax = plt.subplots()
    contour_set = LH_transition_condition.T.plot.contour(levels=[1.0], ax=ax)
    plt.close()

    p = contour_set.collections[0].get_paths()[0]
    v = p.vertices
    LH_separatrix_density = xr.DataArray(v[:, 0], coords=dict(LH_separatrix_density=v[:, 0]))
    LH_separatrix_temp = xr.DataArray(v[:, 1], coords=dict(LH_separatrix_density=v[:, 0]))

    return LH_separatrix_density, LH_separatrix_temp


def interpolate_field_to_LH_curve(
    field: xr.DataArray, LH_separatrix_density: xr.DataArray, LH_separatrix_temp: xr.DataArray
) -> xr.DataArray:
    """Interpolate a 2D field to the points defined by the LH transition curve."""
    interpolator = RectBivariateSpline(
        field.separatrix_density, field.separatrix_temp, field.transpose("separatrix_density", "separatrix_temp")
    )

    interpolated_curve = xr.DataArray(
        interpolator(LH_separatrix_density, LH_separatrix_temp, grid=False),
        coords=dict(LH_separatrix_density=LH_separatrix_density),
    ).pint.quantify(field.pint.units)

    return interpolated_curve  # type:ignore[no-any-return]


def calc_power_crossing_separatrix_in_ion_channel(
    surface_area: Unitfull,
    separatrix_density: Unitfull,
    separatrix_temp: Unitfull,
    lambda_Te: Unitfull,
    chi_i: Unitfull,
    temp_scale_length_ratio: float = 1.0,
) -> Unitfull:
    """Calculate the power crossing the separatrix in the ion channel.

    temp_scale_length_ratio = Ti / Te * lambda_Te / lambda_Ti = L_Te / L_Ti
    """
    chi_i = convert_units(chi_i, ureg.m**2 * ureg.s**-1)

    L_Te = lambda_Te / separatrix_temp
    L_Ti = L_Te / temp_scale_length_ratio

    P_SOL_i = surface_area * separatrix_density * chi_i / L_Ti

    return convert_units(P_SOL_i, ureg.MW)


def calLH_transition_condition_power_required_for_given_Qi_to_Qe(
    P_LH_electron_channel: Unitfull, P_LH_ion_channel: Unitfull, Qi_to_Qe: float = 0.5
) -> Unitfull:
    """Calculate the total power crossing the separatrix, assuming some fixed ratio of Qi/Qe.

    Qi_to_Qe = Qi / Qe
    """
    return np.maximum(P_LH_electron_channel * (1.0 + Qi_to_Qe), P_LH_ion_channel * (1.0 + 1.0 / Qi_to_Qe))
