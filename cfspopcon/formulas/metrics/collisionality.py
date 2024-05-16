"""Calculate the normalized collisionality."""
import numpy as np

from ...algorithm_class import Algorithm
from ...unit_handling import Unitfull, convert_units, ureg, wraps_ufunc


@wraps_ufunc(input_units=dict(electron_density=ureg.m**-3, electron_temp=ureg.eV), return_units=dict(coulomb_log=ureg.dimensionless))
def calc_coulomb_logarithm(electron_density: float, electron_temp: float) -> float:
    """Calculate the Coulomb logarithm, for electron-electron or electron-ion collisions.

    From text on page 6 of :cite:`Verdoolaege_2021`
    """
    return float(30.9 - np.log(electron_density**0.5 * electron_temp**-1.0))


@Algorithm.register_algorithm(return_keys=["nu_star"])
def calc_normalised_collisionality(
    average_electron_density: Unitfull,
    average_electron_temp: Unitfull,
    average_ion_temp: Unitfull,
    q_star: Unitfull,
    major_radius: Unitfull,
    inverse_aspect_ratio: Unitfull,
    z_effective: Unitfull,
) -> Unitfull:
    """Calculate normalized collisionality.

    Equation 1c from :cite:`Verdoolaege_2021`

    Extra factor of ureg.e**2, presumably related to electron_temp**-2 for electron_temp in eV

    Args:
        average_electron_density: [1e19 m^-3] :term:`glossary link<average_electron_density>`
        average_electron_temp: [keV] :term:`glossary link<average_electron_temp>`
        average_ion_temp: [keV] :term:`glossary link<average_ion_temp>`
        q_star: [~] :term:`glossary link<q_star>`
        major_radius: [m] :term:`glossary link<major_radius>`
        inverse_aspect_ratio: [m] :term:`glossary link<inverse_aspect_ratio>`
        z_effective: [~] :term:`glossary link<z_effective>`

    Returns:
         nu_star [~]
    """
    return convert_units(
        ureg.e**4
        / (2.0 * np.pi * 3**1.5 * ureg.epsilon_0**2)
        * calc_coulomb_logarithm(average_electron_density, average_electron_temp)
        * average_electron_density
        * q_star
        * major_radius
        * z_effective
        / (average_ion_temp**2 * inverse_aspect_ratio**1.5),
        ureg.dimensionless,
    )


@Algorithm.register_algorithm(return_keys=["alpha_t"])
def calc_alpha_t(
    separatrix_electron_density: Unitfull,
    separatrix_electron_temp: Unitfull,
    cylindrical_safety_factor: Unitfull,
    major_radius: Unitfull,
    ion_mass: Unitfull,
    z_effective: Unitfull,
    mean_ion_charge_state: Unitfull,
    ion_to_electron_temp_ratio: float = 1.0,
) -> Unitfull:
    """Calculate the turbulence parameter alpha_t.

    Equation 9 from :cite:`Eich_2020`. Compared to this equation, the factor of the
    ion_to_electron_temp_ratio is added following a discussion with T. Eich.


    Args:
        separatrix_electron_density: :term:`glossary link<separatrix_electron_density>`
        separatrix_electron_temp: :term:`glossary link<separatrix_electron_temp>`
        cylindrical_safety_factor: :term:`glossary link<cylindrical_safety_factor>`
        major_radius: :term:`glossary link<major_radius>`
        ion_mass: :term:`glossary link<ion_mass>`
        z_effective: :term:`glossary link<z_effective>`
        mean_ion_charge_state: :term:`glossary link<mean_ion_charge_state>`
        ion_to_electron_temp_ratio: :term:`glossary link<ion_to_electron_temp_ratio>`

    Returns:
        :term:`alpha_t`
    """
    coulomb_log = calc_coulomb_logarithm(electron_density=separatrix_electron_density, electron_temp=separatrix_electron_temp)
    ion_sound_speed = np.sqrt(mean_ion_charge_state * separatrix_electron_temp / ion_mass)
    nu_ei = calc_electron_ion_collision_freq(
        electron_density=separatrix_electron_density,
        electron_temp=separatrix_electron_temp,
        coulomb_log=coulomb_log,
        z_effective=z_effective,
    )

    alpha_t = (
        1.02
        * nu_ei
        / ion_sound_speed
        * (1.0 * ureg.electron_mass / ion_mass)
        * cylindrical_safety_factor**2
        * major_radius
        * (1.0 + ion_to_electron_temp_ratio / mean_ion_charge_state)
    )

    return convert_units(alpha_t, ureg.dimensionless)


@Algorithm.register_algorithm(return_keys=["edge_collisionality"])
def calc_edge_collisionality(
    separatrix_electron_density: Unitfull,
    separatrix_electron_temp: Unitfull,
    cylindrical_safety_factor: Unitfull,
    major_radius: Unitfull,
    ion_mass: Unitfull,
    z_effective: Unitfull,
    mean_ion_charge_state: Unitfull,
    ion_to_electron_temp_ratio: float = 1.0,
) -> Unitfull:
    """Calculate the edge collisionality.

    Equation 7 from :cite:`Faitsch_2023`.

    Args:
        separatrix_electron_density: :term:`glossary link<separatrix_electron_density>`
        separatrix_electron_temp: :term:`glossary link<separatrix_electron_temp>`
        cylindrical_safety_factor: :term:`glossary link<cylindrical_safety_factor>`
        major_radius: :term:`glossary link<major_radius>`
        ion_mass: :term:`glossary link<ion_mass>`
        z_effective: :term:`glossary link<z_effective>`
        mean_ion_charge_state: :term:`glossary link<mean_ion_charge_state>`
        ion_to_electron_temp_ratio: :term:`glossary link<ion_to_electron_temp_ratio>`

    Returns:
        :term:`edge_collisionality`
    """
    alpha_t = calc_alpha_t(
        separatrix_electron_density=separatrix_electron_density,
        separatrix_electron_temp=separatrix_electron_temp,
        cylindrical_safety_factor=cylindrical_safety_factor,
        major_radius=major_radius,
        ion_mass=ion_mass,
        z_effective=z_effective,
        mean_ion_charge_state=mean_ion_charge_state,
        ion_to_electron_temp_ratio=ion_to_electron_temp_ratio,
    )

    return 100.0 * alpha_t / cylindrical_safety_factor


def calc_electron_electron_collision_freq(electron_density: Unitfull, electron_temp: Unitfull, coulomb_log: Unitfull) -> Unitfull:
    """Calculate the electron-electron collision frequency, using equation B1 from from :cite:`Eich_2020`."""
    nu_ee = (
        (4.0 / 3.0)
        * np.sqrt(2.0 * np.pi)
        * electron_density
        * ureg.e**4
        * coulomb_log
        / ((4.0 * np.pi * ureg.epsilon_0) ** 2 * np.sqrt(1.0 * ureg.electron_mass) * electron_temp**1.5)
    )
    return nu_ee


def calc_electron_ion_collision_freq(
    electron_density: Unitfull, electron_temp: Unitfull, coulomb_log: Unitfull, z_effective: Unitfull
) -> Unitfull:
    """Calculate the electron-ion collision frequency, using equation B2 from from :cite:`Eich_2020`."""
    nu_ee = calc_electron_electron_collision_freq(electron_density, electron_temp, coulomb_log)
    z_effective_correction = (1.0 - 0.569) * np.exp(-(((z_effective - 1.0) / 3.25) ** 0.85)) + 0.569

    return nu_ee * z_effective_correction * z_effective
