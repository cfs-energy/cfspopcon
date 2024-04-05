"""Reaction rates (<sigma*v>) as a function of ion temperature."""

from typing import Callable, Union

import numpy as np
from numpy import float64
from numpy.typing import NDArray

from ...named_options import ReactionType


def sigmav_DT(ion_temp_profile: NDArray[float64]) -> NDArray[float64]:
    r"""Deuterium-Tritium reaction.

    Calculate :math:`\langle \sigma v \rangle` for a given characteristic ion energy.
    Formulation from table 1, column S5 in :cite:`hively_convenient_1977`.
    Curvefit was performed for the range of [1,80]keV.

    Args:
        ion_temp_profile: ion temperature profile [keV]

    Returns:
        :math:`\langle \sigma v \rangle` in cm^3/s.
    """
    A = [-21.377692, -25.204054, -7.1013427 * 1e-2, 1.9375451 * 1e-4, 4.9246592 * 1e-6, -3.9836572 * 1e-8]
    r = 0.2935
    sigmav = np.exp(
        A[0] / ion_temp_profile**r
        + A[1]
        + A[2] * ion_temp_profile
        + A[3] * ion_temp_profile**2.0
        + A[4] * ion_temp_profile**3.0
        + A[5] * ion_temp_profile**4.0
    )
    return sigmav  # type: ignore[no-any-return] # [cm^3/s]


def sigmav_DT_BoschHale(ion_temp_profile: NDArray[float64]) -> NDArray[float64]:
    r"""Deuterium-Tritium reaction.

    Calculate :math:`\langle \sigma v \rangle` product for a given characteristic ion energy using Bosch Hale method.

    :func:`sigmav_DT_BoschHale` is more accurate than :func:`sigmav_DT` for ion_temp_profile > ~48.45 keV (estimate based on
    linear interp between errors found at available datapoints).
    Maximum error = 1.4% within range 50-1000 keV from available NRL data.

    Formulation from :cite:`bosch_improved_1992`

    Args:
        ion_temp_profile: ion temperature profile [keV]

    Returns:
        :math:`\langle \sigma v \rangle` in cm^3/s.

    """
    # Bosch Hale coefficients for DT reaction
    C = [0.0, 1.173e-9, 1.514e-2, 7.519e-2, 4.606e-3, 1.35e-2, -1.068e-4, 1.366e-5]
    B_G = 34.3827
    mr_c2 = 1124656

    theta = ion_temp_profile / (
        1
        - (ion_temp_profile * (C[2] + ion_temp_profile * (C[4] + ion_temp_profile * C[6])))
        / (1 + ion_temp_profile * (C[3] + ion_temp_profile * (C[5] + ion_temp_profile * C[7])))
    )
    eta = (B_G**2 / (4 * theta)) ** (1 / 3)
    sigmav = C[1] * theta * np.sqrt(eta / (mr_c2 * ion_temp_profile**3)) * np.exp(-3 * eta)
    return sigmav  # type: ignore[no-any-return] # [cm^3/s]


def sigmav_DD(ion_temp_profile: NDArray[float64]) -> tuple[NDArray[float64], NDArray[float64], NDArray[float64]]:
    r"""Deuterium-Deuterium reaction.

    Calculate :math:`\langle \sigma v \rangle` for a given characteristic ion energy.
    Formulation from column S5, in table 3 and 4 in :cite:`hively_convenient_1977`.
    Curvefit was performed for the range of [1,80]keV.

    Args:
        ion_temp_profile: ion temperature profile [keV]

    Returns:
        :math:`\langle \sigma v \rangle` tuple (total, D(d,p)T, D(d,n)3He) in cm^3/s.
    """
    a_1 = [
        -15.511891,
        -35.318711,
        -1.2904737 * 1e-2,
        2.6797766 * 1e-4,
        -2.9198685 * 1e-6,
        1.2748415 * 1e-8,
    ]  # For D(d,p)T
    r_1 = 0.3735
    a_2 = [
        -15.993842,
        -35.017640,
        -1.3689787 * 1e-2,
        2.7089621 * 1e-4,
        -2.9441547 * 1e-6,
        1.2841202 * 1e-8,
    ]  # For D(d,n)3He
    r_2 = 0.3725
    # Ti in units of keV, sigmav in units of cm^3/s
    sigmav_1: NDArray[float64] = np.exp(
        a_1[0] / ion_temp_profile**r_1
        + a_1[1]
        + a_1[2] * ion_temp_profile
        + a_1[3] * ion_temp_profile**2.0
        + a_1[4] * ion_temp_profile**3.0
        + a_1[5] * ion_temp_profile**4.0
    )
    sigmav_2: NDArray[float64] = np.exp(
        a_2[0] / ion_temp_profile**r_2
        + a_2[1]
        + a_2[2] * ion_temp_profile
        + a_2[3] * ion_temp_profile**2.0
        + a_2[4] * ion_temp_profile**3.0
        + a_2[5] * ion_temp_profile**4.0
    )
    sigmav_tot: NDArray[float64] = sigmav_1 + sigmav_2
    return sigmav_tot, sigmav_1, sigmav_2  # [cm^3/s]


def sigmav_DD_BoschHale(ion_temp_profile: NDArray[float64]) -> tuple[NDArray[float64], NDArray[float64], NDArray[float64]]:
    r"""Deuterium-Deuterium reaction.

    Calculate :math:`\langle \sigma v \rangle` product for a given characteristic ion energy using Bosch Hale method.

    Function tested on available data at [1, 2, 5, 10, 20, 50, 100] keV.
    Maximum error = 3.8% within range 5-50 keV and increases significantly outside of [5, 50] keV.

    Uses DD cross section formulation from :cite:`bosch_improved_1992`.

    Other form in :cite:`langenbrunner_analytic_2017`.

    Args:
        ion_temp_profile: ion temperature profile [keV]

    Returns:
        :math:`\langle \sigma v \rangle` tuple (total, D(d,p)T, D(d,n)3He) in cm^3/s.
    """
    # For D(d,n)3He
    cBH_1 = [((31.3970**2) / 4.0) ** (1.0 / 3.0), 5.65718e-12, 3.41e-03, 1.99e-03, 0, 1.05e-05, 0, 0]  # 3.72e-16,

    mc2_1 = 937814.0

    # For D(d,p)T
    cBH_2 = [((31.3970**2) / 4.0) ** (1.0 / 3.0), 5.43360e-12, 5.86e-03, 7.68e-03, 0, -2.96e-06, 0, 0]  # 3.57e-16,

    mc2_2 = 937814.0

    thetaBH_1 = ion_temp_profile / (
        1
        - (
            (cBH_1[2] * ion_temp_profile + cBH_1[4] * ion_temp_profile**2 + cBH_1[6] * ion_temp_profile**3)
            / (1 + cBH_1[3] * ion_temp_profile + cBH_1[5] * ion_temp_profile**2 + cBH_1[7] * ion_temp_profile**3)
        )
    )

    thetaBH_2 = ion_temp_profile / (
        1
        - (
            (cBH_2[2] * ion_temp_profile + cBH_2[4] * ion_temp_profile**2 + cBH_2[6] * ion_temp_profile**3)
            / (1 + cBH_2[3] * ion_temp_profile + cBH_2[5] * ion_temp_profile**2 + cBH_2[7] * ion_temp_profile**3)
        )
    )

    etaBH_1: float = cBH_1[0] / (thetaBH_1 ** (1.0 / 3.0))
    etaBH_2: float = cBH_2[0] / (thetaBH_2 ** (1.0 / 3.0))

    sigmav_1: NDArray[float64] = cBH_1[1] * thetaBH_1 * np.sqrt(etaBH_1 / (mc2_1 * (ion_temp_profile**3.0))) * np.exp(-3.0 * etaBH_1)
    sigmav_2: NDArray[float64] = cBH_2[1] * thetaBH_2 * np.sqrt(etaBH_2 / (mc2_2 * (ion_temp_profile**3.0))) * np.exp(-3.0 * etaBH_2)
    sigmav_tot: NDArray[float64] = sigmav_1 + sigmav_2

    return sigmav_tot, sigmav_1, sigmav_2  # [cm^3/s]


def sigmav_DHe3(ion_temp_profile: NDArray[float64]) -> NDArray[float64]:
    r"""Deuterium-Helium-3 reaction.

    Calculate :math:`\langle \sigma v \rangle` for a given characteristic ion energy.

    Function tested on available data at [1, 2, 5, 10, 20, 50, 100] keV.
    Maximum error = 8.4% within range 2-100 keV and should not be used outside range [2, 100] keV.

    Uses DD cross section formulation :cite:`bosch_improved_1992`.

    Args:
        ion_temp_profile: ion temperature profile [keV]

    Returns:
        :math:`\langle \sigma v \rangle` in cm^3/s.
    """
    # For He3(d,p)4He
    cBH_1 = [
        ((68.7508**2) / 4.0) ** (1.0 / 3.0),
        5.51036e-10,  # 3.72e-16,
        6.41918e-03,
        -2.02896e-03,
        -1.91080e-05,
        1.35776e-04,
        0,
        0,
    ]

    mc2_1 = 1124572.0

    thetaBH_1 = ion_temp_profile / (
        1
        - (
            (cBH_1[2] * ion_temp_profile + cBH_1[4] * ion_temp_profile**2 + cBH_1[6] * ion_temp_profile**3)
            / (1 + cBH_1[3] * ion_temp_profile + cBH_1[5] * ion_temp_profile**2 + cBH_1[7] * ion_temp_profile**3.0)
        )
    )

    etaBH_1: float = cBH_1[0] / (thetaBH_1 ** (1.0 / 3.0))

    sigmav: NDArray[float64] = cBH_1[1] * thetaBH_1 * np.sqrt(etaBH_1 / (mc2_1 * (ion_temp_profile**3.0))) * np.exp(-3.0 * etaBH_1)

    return sigmav  # [cm^3/s]


def sigmav_pB11(ion_temp_profile: NDArray[float64]) -> NDArray[float64]:
    r"""Proton (hydrogen)-Boron11 reaction.

    Calculate :math:`\langle \sigma v \rangle` for a given characteristic ion energy.

    Uses cross section from Nevins and Swain :cite:`nevins_thermonuclear_2000`.
    Updated cross sections in :cite:`sikora_new_2016`, and :cite:`putvinski_fusion_2019` are not in analytic form.

    Args:
        ion_temp_profile: characteristic ion energy in keV.

    Returns:
        :math:`\langle \sigma v \rangle` in cm^3/s.
    """
    # High temperature (T>60 keV)
    # For B11(p,alpha)alpha,alpha
    cBH_1 = [
        ((22589.0) / 4.0) ** (1.0 / 3.0),
        4.4467e-14,
        -5.9357e-02,
        2.0165e-01,
        1.0404e-03,
        2.7621e-03,
        -9.1653e-06,
        9.8305e-07,
    ]

    mc2_1 = 859526.0

    thetaBH_1 = ion_temp_profile / (
        1
        - (
            (cBH_1[2] * ion_temp_profile + cBH_1[4] * ion_temp_profile**2 + cBH_1[6] * ion_temp_profile**3)
            / (1 + cBH_1[3] * ion_temp_profile + cBH_1[5] * ion_temp_profile**2 + cBH_1[7] * ion_temp_profile**3)
        )
    )

    etaBH_1 = cBH_1[0] / (thetaBH_1 ** (1.0 / 3.0))

    sigmavNRhigh = (
        cBH_1[1] * thetaBH_1 * np.sqrt(etaBH_1 / (mc2_1 * (ion_temp_profile**3.0))) * np.exp(-3.0 * etaBH_1)
    ) * 1e6  # m3 to cm3

    # Low temperature (T<60 keV)
    E0 = ((17.81) ** (1.0 / 3.0)) * (ion_temp_profile ** (2.0 / 3.0))

    deltaE0 = 4.0 * np.sqrt(ion_temp_profile * E0 / 3.0)

    tau = (3.0 * E0) / ion_temp_profile

    Mp = 1.0  # *1.67e-27
    MB = 11.0  # *1.67e-27

    Mr = (Mp * MB) / (Mp + MB)

    C0 = 197.000 * 1e-25  # MeVb to kev/m^2
    C1 = 0.240 * 1e-25  # MeVb to kev/m^2
    C2 = 0.000231 * 1e-25  # MeVb to kev/m^2

    Seff = (
        C0 * (1 + (5.0 / (12.0 * tau)))
        + C1 * (E0 + (35.0 / 36.0) * ion_temp_profile)
        + C2 * (E0**2.0 + (89.0 / 36.0) * E0 * ion_temp_profile)
    )

    sigmavNRlow: NDArray[float64] = (
        np.sqrt(2 * ion_temp_profile / Mr) * ((deltaE0 * Seff) / (ion_temp_profile ** (2.0))) * np.exp(-tau)
    ) * 1e6  # m3 to cm3
    # 148 keV resonance
    sigmavR: NDArray[float64] = (
        (5.41e-21) * ((1.0 / ion_temp_profile) ** (3.0 / 2.0)) * np.exp(-148.0 / ion_temp_profile)
    ) * 1e6  # m3 to cm3
    sigmav: NDArray[float64] = sigmavNRhigh

    for i in range(len(ion_temp_profile)):
        if ion_temp_profile[i] < 60.0:  # keV
            sigmav[i] = sigmavNRlow[i] + sigmavR[i]
        elif (ion_temp_profile[i] > 60.0) and (ion_temp_profile[i] < 130):  # keV
            sigmav[i] = sigmavNRhigh[i] + sigmavR[i]

    return sigmav  # [cm^3/s]

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
