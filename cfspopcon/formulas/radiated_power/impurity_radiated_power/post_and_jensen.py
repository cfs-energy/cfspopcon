"""Calculate the radiated power due to impurities, according to an analytical fitted curve."""

import warnings

import numpy as np
from numpy import float64
from numpy.polynomial.polynomial import polyval
from numpy.typing import NDArray

from ....algorithm_class import Algorithm
from ....named_options import AtomicSpecies
from ....unit_handling import ureg, wraps_ufunc
from ...geometry.volume_integral import integrate_profile_over_volume


@Algorithm.register_algorithm(return_keys=["P_rad_impurity"])
@wraps_ufunc(
    return_units=dict(radiated_power=ureg.MW),
    input_units=dict(
        rho=ureg.dimensionless,
        electron_temp_profile=ureg.keV,
        electron_density_profile=ureg.n19,
        impurity_concentration=ureg.dimensionless,
        impurity_species=None,
        plasma_volume=ureg.m**3,
    ),
    input_core_dims=[("dim_rho",), ("dim_rho",), ("dim_rho",), (), (), ()],
)
def calc_impurity_radiated_power_post_and_jensen(
    rho: NDArray[float64],
    electron_temp_profile: NDArray[float64],
    electron_density_profile: NDArray[float64],
    impurity_concentration: float,
    impurity_species: AtomicSpecies,
    plasma_volume: float,
) -> float:
    """Calculation of radiated power using Post & Jensen 1977.

    Radiation fits to the Post & Jensen cooling curves, which use the
    coronal equilibrium model with data in :cite:`post_steady_1977`.

    Args:
        rho: [~] :term:`glossary link<rho>`
        electron_temp_profile: [keV] :term:`glossary link<electron_temp_profile>`
        electron_density_profile: [1e19 m^-3] :term:`glossary link<electron_density_profile>`
        impurity_species: [] :term:`glossary link<impurity_species>`
        impurity_concentration: [~] :term:`glossary link<impurity_concentration>`
        plasma_volume: [m^3] :term:`glossary link<plasma_volume>`

    Returns:
         [MW] Estimated radiation power due to this impurity
    """
    impurity_Z = impurity_species.value

    zimp = np.array([2, 4, 5, 6, 8, 18, 22, 26, 28, 42, 74])

    if impurity_Z not in zimp:  # pragma: no cover
        warnings.warn(f"Post & Jenson radiation calculation not supported for impurity with Z={impurity_Z}", stacklevel=3)
        return np.nan

    # get the index of the impurity
    iz = np.nonzero(zimp == impurity_Z)[0][0]

    # supported minimum temperature for each supported impurity above
    Tmin = np.array([0.002, 0.002, 0.002, 0.003, 0.005, 0.03, 0.02, 0.02, 0.03, 0.06, 0.1])

    # If trying to evaluate for a temperature outside of the given range, assume nearest neighbor
    # and throw a warning
    if any(electron_temp_profile < Tmin[iz]) or any(electron_temp_profile > 100):  # pragma: no cover
        warnings.warn(
            f"Post 1977 line radiation calculation is only valid for Z={impurity_Z} between {Tmin[iz]}-100keV. Using nearest neighbor extrapolation.",
            stacklevel=3,
        )
    electron_temp_profile = np.maximum(electron_temp_profile, Tmin[iz])
    electron_temp_profile = np.minimum(electron_temp_profile, 100)

    temperature_bin_borders = np.array([0.0, 0.02, 0.2, 2.0, 20.0, 100.0])

    # A_i coefficients for the first temperature bin, for the 10 supported impurities
    # e.g. radc1[0][3] would be the A(0) coefficient for the first temperature
    # bin for Carbon (zimp[3]==6).
    # radc1[:][0] would be the [A_0, A_1..., A_5] coefficients for Helium.
    radc1 = np.array(
        [
            [144.1278, -342.5149, -1508.695, 1965.3, 652.374, 0, 0, 0, 0, 0, 0],
            [294.0867, -947.126, -3512.267, 4572.039, 1835.499, 0, 0, 0, 0, 0, 0],
            [176.1164, -1035.776, -3286.123, 4159.59, 1984.266, 0, 0, 0, 0, 0, 0],
            [33.8743, -538.2415, -1520.07, 1871.56, 1059.846, 0, 0, 0, 0, 0, 0],
            [-3.075936, -134.9198, -347.0698, 417.3889, 280.0476, 0, 0, 0, 0, 0, 0],
            [-1.204179, -13.19063, -31.27689, 36.99382, 29.33792, 0, 0, 0, 0, 0, 0],
        ],
        dtype=float64,
    )

    # Coefficients for 2nd temperature bin
    radc2 = np.array(
        [
            [-22.7421, -34.29832, -63.7016, 74.67599, -55.15118, -20.53043, 23.91331, -27.52599, -12.03248, -139.1054, 5.340828],
            [-0.7402954, -58.04948, -215.6758, 454.9038, -154.3956, -2.834287, 183.3595, -39.08228, 32.53908, -649.3335, 156.0876],
            [-2.177691, -103.257, -430.8101, 837.2937, -248.992, 15.06902, 301.9617, -64.69423, 67.90773, -1365.838, 417.1704],
            [-2.426768, -88.53015, -422.2842, 740.2515, -180.8154, 35.17177, 237.6019, -55.55048, 65.29924, -1406.464, 550.2576],
            [-1.026211, -35.30521, -200.8412, 314.7607, -57.64175, 24.00122, 90.49792, -24.05568, 29.73465, -708.6213, 356.7583],
            [-0.1798547, -5.13446, -36.87482, 51.64578, -6.149181, 5.072723, 13.4509, -4.09316, 5.271279, -140.0571, 90.42786],
        ]
    )

    # Coefficients for 3rd temperature bin
    radc3 = np.array(
        [
            [-22.54156, -21.77747, -21.47874, -21.20151, -20.68816, -19.65204, -18.99097, -18.34973, -18.30482, -17.72591, -17.23894],
            [0.350319, 0.04617764, -0.15653, -0.3668933, -0.7482238, -0.1172763, -3.403261, -1.252028, -0.003319243, -1.058217, 0.05423752],
            [0.1210755, 0.4411196, 0.6181287, 0.7295099, 0.7390959, 7.83322, 1.43983, -7.533115, -3.332313, -3.583172, -1.22107],
            [-0.1171573, -0.2972147, -0.2477378, -0.1944827, -0.672159, -6.351577, 17.35576, -3.289693, -11.12798, 1.660089, 0.4411812],
            [0.08237547, 0.04526295, -0.1060488, -0.1263576, 1.338345, -30.58849, 0.2804832, 28.66739, 0.1053073, 8.565372, -4.485821],
            [0.04361719, 0.1266794, -0.04537644, -0.1491027, 3.734628, -15.28534, -19.43971, 28.30249, 9.448907, 4.532909, -7.836137],
        ]
    )

    # Coefficients for 4th temperature bin
    radc4 = np.array(
        [
            [-22.58311, -21.77388, -21.47337, -21.21979, -20.66766, -19.74883, -19.29037, -16.71042, -16.97678, -13.85096, -14.7488],
            [0.6858961, 0.0147862, -0.1829426, -0.2346986, -0.955559, 2.964839, -3.260377, -16.46143, -9.49547, -36.78452, -14.39542],
            [-0.8628176, 0.5259617, 0.6678447, 0.4093794, 1.467982, -8.829391, 14.54427, 37.66238, 11.09362, 114.0587, 21.05855],
            [1.205242, -0.381643, -0.3864809, 0.07874548, -0.9822488, 9.791004, -23.83997, -39.4408, 0.04045904, -163.5634, -4.394746],
            [-0.738631, 0.15834, 0.116592, -0.1841379, 0.4171964, -4.960018, 16.42804, 19.18529, -6.521934, 107.626, -11.06006],
            [0.168653, -0.02891062, -0.01400226, 0.05590744, -0.08244216, 0.9820032, -4.084697, -3.509238, 2.654915, -26.42488, 5.616985],
        ]
    )

    # Coefficients for 5th temperature bin
    radc5 = np.array(
        [
            [-17.30458, -20.31496, -24.46008, -24.76796, -27.80602, -21.17935, -13.4178, -24.53957, -28.64081, 39.92683, -262.426],
            [-16.24615, -4.2026, 9.26496, 9.408181, 21.46056, 5.191481, -16.75967, 17.95222, 29.99289, -175.7093, 712.5586],
            [21.00786, 5.289472, -11.10684, -9.657446, -26.65906, -7.439717, 18.4332, -23.5636, -37.26082, 207.4927, -825.0168],
            [-13.12075, -3.002456, 6.837004, 4.999161, 16.70831, 4.969023, -10.33234, 14.84503, 22.5806, -121.4589, 474.2407],
            [4.06935, 0.8528627, -2.065106, -1.237382, -5.191943, -1.55318, 2.96053, -4.542323, -6.716598, 35.31804, -135.5175],
            [-0.5009445, -0.09687936, 0.2458526, 0.116061, 0.6410295, 0.1877047, -0.3423194, 0.5477462, 0.7911687, -4.083832, 15.41889],
        ]
    )

    radc = np.array([radc1, radc2, radc3, radc4, radc5])

    Tlog = np.log10(electron_temp_profile)
    log10_Lz = np.zeros(electron_temp_profile.size)

    for i in range(len(radc)):
        it = np.nonzero((electron_temp_profile >= temperature_bin_borders[i]) & (electron_temp_profile < temperature_bin_borders[i + 1]))[0]
        if it.size > 0:
            log10_Lz[it] = polyval(Tlog[it], radc[i, :, iz])

    radrate = 10.0**log10_Lz
    radrate[np.isnan(radrate)] = 0
    radrate /= 1e13  # convert from erg cm^3 -> J m^3  (erg == 1e-7 J)

    # 1e38 factor to account for the fact that our n_e values are electron_density_profile values
    qRad = radrate * electron_density_profile * electron_density_profile * impurity_concentration * 1e38  # W / (m^3 s)
    radiated_power = integrate_profile_over_volume.unitless_func(qRad, rho, plasma_volume)  # [W]
    return float(radiated_power) / 1e6  # MW
