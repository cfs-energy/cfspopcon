"""Reaction energies and power densities."""

from typing import Any

from numpy import float64
from numpy.typing import NDArray
from scipy import constants  # type: ignore[import-untyped]


def reaction_energy_DT(
    sigmav: NDArray[float64], heavier_fuel_species_fraction: float
) -> tuple[float, float, float, NDArray[float64], NDArray[float64], NDArray[float64]]:
    r"""Deuterium-Tritium reaction.

    Calculate reaction energies and power density values.

    Args:
        sigmav: :math:`\langle \sigma v \rangle` product in cm^3/s.
        heavier_fuel_species_fraction: n_Tritium / (n_Tritium + n_Deuterium) number fraction.

    Returns:
        Tuple of reaction energies and corresponding power densities.
    """
    rxn_energy: float = 17.6 * constants.value("electron volt")  # [MJ]
    rxn_energy_neut: float = rxn_energy * (4.0 / 5.0)  # [MJ]
    rxn_energy_charged: float = rxn_energy * (1.0 / 5.0)  # [MJ]
    convert_volume: float = 1e-6  # [m^3/cm^3]
    number_power_dens: NDArray[float64] = (
        sigmav * rxn_energy * convert_volume * heavier_fuel_species_fraction * (1.0 - heavier_fuel_species_fraction)
    )
    number_power_dens_neut: NDArray[float64] = (
        sigmav * rxn_energy_neut * convert_volume * heavier_fuel_species_fraction * (1.0 - heavier_fuel_species_fraction)
    )
    number_power_dens_charged: NDArray[float64] = (
        sigmav * rxn_energy_charged * convert_volume * heavier_fuel_species_fraction * (1.0 - heavier_fuel_species_fraction)
    )

    # Units: [MJ], [MW*m^3]
    return rxn_energy, rxn_energy_neut, rxn_energy_charged, number_power_dens, number_power_dens_neut, number_power_dens_charged


def reaction_energy_DD(
    sigmav: tuple[
        NDArray[float64],
        NDArray[float64],
        NDArray[float64],
    ],
    **_: Any,
) -> tuple[NDArray[float64], float, NDArray[float64], NDArray[float64], NDArray[float64], NDArray[float64]]:
    r"""Deuterium-Deuterium reaction.

    Calculate reaction energies and power density values.

    Args:
        sigmav: :math:`\langle \sigma v \rangle` product in cm^3/s.
        _: Unused placeholder to enable unified call syntax with e.g. :func:`reaction_energy_DT`.

    Returns:
        Tuple of reaction energies and corresponding power densities.
    """
    sigmav_tot, sigmav_1, sigmav_2 = sigmav
    path_1_energy: float = (1.01 + 3.02) * constants.value("electron volt")  # MJ, D+D -> p+T
    path_2_energy: float = (0.82 + 2.45) * constants.value("electron volt")  # MJ, D+D -> n+He3
    rxn_energy: NDArray[float64] = (path_1_energy * sigmav_1 + path_2_energy * sigmav_2) / sigmav_tot
    rxn_energy_neut: float = path_2_energy * (3.0 / 4.0)
    rxn_energy_charged: NDArray[float64] = (path_1_energy * sigmav_1 + (path_2_energy * (1.0 / 4.0)) * sigmav_2) / sigmav_tot

    # So number_power_dens*electron_density_profile**2 = power_dens [MW/m^3] no need to divide since nD=ne
    convert_volume = 1e-6  # m^3/cm^3
    number_power_dens: NDArray[float64] = (sigmav_1 * path_1_energy + sigmav_2 * path_2_energy) * convert_volume
    number_power_dens_neut: NDArray[float64] = sigmav_2 * rxn_energy_neut * convert_volume
    number_power_dens_charged: NDArray[float64] = (sigmav_1 * path_1_energy + sigmav_2 * path_2_energy * (1.0 / 4.0)) * convert_volume

    # Units: [MJ], [MW/m^3]
    return rxn_energy, rxn_energy_neut, rxn_energy_charged, number_power_dens, number_power_dens_neut, number_power_dens_charged


def reaction_energy_DHe3(
    sigmav: NDArray[float64], heavier_fuel_species_fraction: float
) -> tuple[float, float, float, NDArray[float64], NDArray[float64], NDArray[float64]]:
    r"""Deuterium-Helium 3 reaction.

    Calculate reaction energies and power density values.

    Args:
        sigmav: :math:`\langle \sigma v \rangle` product in cm^3/s.
        heavier_fuel_species_fraction: n_heavier / (n_heavier + n_lighter) number fraction.

    Returns:
        Tuple of reaction energies and corresponding power densities.
    """
    rxn_energy: float = 18.3 * constants.value("electron volt")  # MJoules
    rxn_energy_neut: float = 0.0
    rxn_energy_charged: float = rxn_energy
    convert_volume: float = 1e-6  # m^3/cm^3
    number_power_dens: NDArray[float64] = (
        sigmav * rxn_energy * convert_volume * heavier_fuel_species_fraction * (1.0 - heavier_fuel_species_fraction)
    )
    number_power_dens_neut: NDArray[float64] = (
        sigmav * rxn_energy_neut * convert_volume * heavier_fuel_species_fraction * (1.0 - heavier_fuel_species_fraction)
    )
    number_power_dens_charged: NDArray[float64] = (
        sigmav * rxn_energy_charged * convert_volume * heavier_fuel_species_fraction * (1.0 - heavier_fuel_species_fraction)
    )

    # Units: [MJ], [MW/m^3]
    return rxn_energy, rxn_energy_neut, rxn_energy_charged, number_power_dens, number_power_dens_neut, number_power_dens_charged


def reaction_energy_pB11(
    sigmav: NDArray[float64], heavier_fuel_species_fraction: float
) -> tuple[float, float, float, NDArray[float64], NDArray[float64], NDArray[float64]]:
    r"""Proton (hydrogen)-Boron-11 reaction.

    Calculate reaction energies and power density values.

    Args:
        sigmav: :math:`\langle \sigma v \rangle` product in cm^3/s.
        heavier_fuel_species_fraction: n_heavier / (n_heavier + n_lighter) number fraction.

    Returns:
        Tuple of reaction energies and corresponding power densities.
    """
    rxn_energy: float = 8.7 * constants.value("electron volt")  # MJoules
    rxn_energy_neut: float = 0.0
    rxn_energy_charged: float = rxn_energy
    # This is accurate to within 1%
    convert_volume: float = 1e-6  # m^3/cm^3
    number_power_dens: NDArray[float64] = (
        sigmav * rxn_energy * convert_volume * heavier_fuel_species_fraction * (1.0 - heavier_fuel_species_fraction)
    )
    number_power_dens_neut: NDArray[float64] = (
        sigmav * rxn_energy_neut * convert_volume * heavier_fuel_species_fraction * (1.0 - heavier_fuel_species_fraction)
    )
    number_power_dens_charged: NDArray[float64] = (
        sigmav * rxn_energy_charged * convert_volume * heavier_fuel_species_fraction * (1.0 - heavier_fuel_species_fraction)
    )

    # Units: [MJ], [MW/m^3]
    return rxn_energy, rxn_energy_neut, rxn_energy_charged, number_power_dens, number_power_dens_neut, number_power_dens_charged
