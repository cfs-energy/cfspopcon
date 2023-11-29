"""Routines to calculate the alpha_t turbulence parameter."""
import numpy as np

from ...unit_handling import Quantity, Unitfull, convert_units, ureg, wraps_ufunc


def calc_alpha_t(
    electron_density: Unitfull,
    electron_temperature: Unitfull,
    edge_safety_factor: Unitfull,
    major_radius: Unitfull,
    ion_mass: Unitfull,
    Zeff: Unitfull,
    Z: Unitfull,
    ion_to_electron_temperature_ratio: float = 1.0,
) -> Unitfull:
    """Calculate the turbulence parameter alpha_t.

    Equivalent to alpha_t = C * omega_B
    where C = normalized collisionality and omega_B = curvature drive term
    """
    ne = electron_density
    Te = electron_temperature

    q = edge_safety_factor
    R = major_radius
    m_e = Quantity(1.0, ureg.electron_mass)
    m_i = ion_mass
    tau_i = ion_to_electron_temperature_ratio

    f_Zeff = (1.0 - 0.569) * np.exp(-(((Zeff - 1.0) / 3.25) ** 0.85)) + 0.569
    coulomb_log = calc_coulomb_log(ne, Te)

    c1 = 1.02 * np.sqrt(2) * ureg.e**4 * coulomb_log * ne / (12.0 * np.pi**1.5 * ureg.epsilon_0**2 * Te**2)

    alpha_t = q**2 * R * np.sqrt(m_e / m_i) * c1 * np.sqrt(Z) * (1 + tau_i / Z) * Zeff * f_Zeff
    return convert_units(alpha_t, ureg.dimensionless)


@wraps_ufunc(input_units=dict(ne=ureg.cm**-3, Te=ureg.eV), return_units=dict(lambda_ee=ureg.dimensionless))
def calc_coulomb_log(ne: float, Te: float) -> float:
    """Compute the Coulomb logarithm for thermal electron-electron collisions.

    From https://farside.ph.utexas.edu/teaching/plasma/Plasma/node39.html
    """
    return 24.0 - np.log(ne**0.5 * Te**-1)  # type:ignore[no-any-return]


@wraps_ufunc(
    input_units=dict(
        electron_density=ureg.m**-3,
        electron_temperature=ureg.eV,
        edge_safety_factor=ureg.dimensionless,
        major_radius=ureg.m,
        ion_mass=ureg.amu,
        Zeff=ureg.dimensionless,
        Z=ureg.dimensionless,
        ion_to_electron_temperature_ratio=ureg.dimensionless,
    ),
    return_units=dict(alpha_t=ureg.dimensionless),
)
def calc_alpha_t_with_fixed_coulomb_log(
    electron_density: Unitfull,
    electron_temperature: Unitfull,
    edge_safety_factor: Unitfull,
    major_radius: Unitfull,
    ion_mass: Unitfull,
    Zeff: Unitfull,
    Z: Unitfull,
    ion_to_electron_temperature_ratio: float = 1.0,
) -> Unitfull:
    """Calculate the turbulence parameter alpha_t, keeping the coulomb logarithm fixed to 13.7.

    Approximately equivalent to equation 12 from :cite:`Eich_2021`, but keeping the Zeff function.
    """
    ne = electron_density
    Te = electron_temperature

    q = edge_safety_factor
    R = major_radius
    A = ion_mass
    tau_i = ion_to_electron_temperature_ratio

    f_Zeff = (1.0 - 0.569) * np.exp(-(((Zeff - 1.0) / 3.25) ** 0.85)) + 0.569

    alpha_t = 2.22e-18 * R * q**2 * ne / Te**2 * Zeff * (1 + tau_i / Z) * np.sqrt(Z / A) * f_Zeff
    return alpha_t
