"""Data for different fusion reactions."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from cfspopcon.unit_handling import Quantity, Unitfull, convert_units, ureg, wraps_ufunc


class FusionReaction:
    """A base class for different fusion reactions."""

    instances: ClassVar[dict[str, FusionReaction]] = dict()

    def __init__(self) -> None:
        """Records children classes in instances."""
        self.instances[self.__class__.__name__] = self


class DTFusionBoschHale(FusionReaction):
    """Deuterium-Tritium reaction using Bosch-Hale cross-section."""

    def __init__(self) -> None:
        """Sets the reaction energies for the Deuterium-Tritium reaction."""
        super().__init__()
        self.energy_per_reaction = Quantity(17.6, ureg.MeV)
        self.energy_to_neutrals_per_reaction = self.energy_per_reaction * 4.0 / 5.0
        self.energy_to_charged_per_reaction = self.energy_per_reaction * 1.0 / 5.0

    @staticmethod
    @wraps_ufunc(input_units=dict(ion_temp=ureg.keV), return_units=dict(sigmav=ureg.cm**3 / ureg.s))
    def calc_rate_coefficient(ion_temp: float) -> float:
        r"""Calculate :math:`\\langle \\sigma v \rangle` for a given ion temperature.

        Cross-section from :cite:`bosch_improved_1992`

        Args:
            ion_temp: [keV]

        Returns:
            :math:`\\langle \\sigma v \rangle` [cm^3/s]
        """
        C = [0.0, 1.173e-9, 1.514e-2, 7.519e-2, 4.606e-3, 1.35e-2, -1.068e-4, 1.366e-5]
        B_G = 34.3827
        mr_c2 = 1124656

        theta = ion_temp / (
            1 - (ion_temp * (C[2] + ion_temp * (C[4] + ion_temp * C[6]))) / (1 + ion_temp * (C[3] + ion_temp * (C[5] + ion_temp * C[7])))
        )
        eta = (B_G**2 / (4 * theta)) ** (1 / 3)
        sigmav: float = C[1] * theta * np.sqrt(eta / (mr_c2 * ion_temp**3)) * np.exp(-3 * eta)

        return sigmav

    def calc_average_fuel_ion_mass(self, heavier_fuel_species_fraction: Unitfull) -> Unitfull:
        """Calculate the average mass of the fuel ions.

        Args:
            heavier_fuel_species_fraction: n_heavier / (n_heavier + n_lighter) number fraction.

        Returns:
            :term:`fuel_average_mass_number` [amu]
        """
        average_fuel_ion_mass = 2.0 * (1 - heavier_fuel_species_fraction) + 3.0 * heavier_fuel_species_fraction
        return average_fuel_ion_mass * ureg.amu

    def calc_energy_per_reaction(self) -> Unitfull:
        """Returns the total energy per reaction."""
        return convert_units(self.energy_per_reaction, ureg.MJ)

    def calc_energy_to_neutrals_per_reaction(self) -> Unitfull:
        """Returns the energy going to uncharged species per reaction."""
        return convert_units(self.energy_to_neutrals_per_reaction, ureg.MJ)

    def calc_energy_to_charged_per_reaction(self) -> Unitfull:
        """Returns the energy going to charged species per reaction."""
        return convert_units(self.energy_to_charged_per_reaction, ureg.MJ)

    def calc_power_density(self, ion_temp: Unitfull, heavier_fuel_species_fraction: Unitfull) -> Unitfull:
        """Returns the total power density."""
        sigmav = self.calc_rate_coefficient(ion_temp)
        fuel_ratio = heavier_fuel_species_fraction * (1.0 - heavier_fuel_species_fraction)

        power_density = sigmav * self.energy_per_reaction * fuel_ratio

        return convert_units(power_density, ureg.MW * ureg.m**3)

    def calc_power_density_to_neutrals(self, ion_temp: Unitfull, heavier_fuel_species_fraction: Unitfull) -> Unitfull:
        """Returns the power density going to uncharged species."""
        return (4.0 / 5.0) * self.calc_power_density(ion_temp, heavier_fuel_species_fraction)

    def calc_power_density_to_charged(self, ion_temp: Unitfull, heavier_fuel_species_fraction: Unitfull) -> Unitfull:
        """Returns the power density going to charged species."""
        return (1.0 / 5.0) * self.calc_power_density(ion_temp, heavier_fuel_species_fraction)


class DTFusionHively(DTFusionBoschHale):
    """Deuterium-Tritium reaction using Hively cross-section."""

    @staticmethod
    @wraps_ufunc(input_units=dict(ion_temp=ureg.keV), return_units=dict(sigmav=ureg.cm**3 / ureg.s))
    def calc_rate_coefficient(ion_temp: float) -> float:
        r"""Calculate :math:`\\langle \\sigma v \rangle` for a given ion temperature.

        Cross-section from :cite:`hively_convenient_1977`

        Args:
            ion_temp: [keV]

        Returns:
            :math:`\\langle \\sigma v \rangle` [cm^3/s]
        """
        A = [-21.377692, -25.204054, -7.1013427 * 1e-2, 1.9375451 * 1e-4, 4.9246592 * 1e-6, -3.9836572 * 1e-8]
        r = 0.2935
        sigmav: float = np.exp(
            A[0] / ion_temp**r + A[1] + A[2] * ion_temp + A[3] * ion_temp**2.0 + A[4] * ion_temp**3.0 + A[5] * ion_temp**4.0
        )
        return sigmav


class DDFusionBoschHale(FusionReaction):
    """Deuterium-Deuterium reaction using Bosch-Hale cross-section."""

    def __init__(self) -> None:
        """Sets the reaction energies for the Deuterium-Deuterium reaction."""
        super().__init__()

        self.energy_per_DD_to_pT_reaction = Quantity(1.01 + 3.02, ureg.MeV)
        self.energy_per_DD_to_nHe3_reaction = Quantity(0.82 + 2.45, ureg.MeV)

        self.energy_to_neutrals_per_DD_to_nHe3_reaction = Quantity(2.45, ureg.MeV)
        self.energy_to_charged_per_DD_to_nHe3_reaction = Quantity(0.82, ureg.MeV)

        self.energy_to_neutrals_per_DD_to_pT_reaction = Quantity(0.0, ureg.MeV)
        self.energy_to_charged_per_DD_to_pT_reaction = Quantity(1.01 + 3.02, ureg.MeV)

    @staticmethod
    @wraps_ufunc(
        input_units=dict(ion_temp=ureg.keV),
        return_units=dict(sigmav_combined=ureg.cm**3 / ureg.s, sigmav_DD_to_pT=ureg.cm**3 / ureg.s, sigmav_DD_to_nHe3=ureg.cm**3 / ureg.s),
        output_core_dims=[(), (), ()],
    )
    def calc_rate_coefficient(ion_temp: float) -> tuple[float, float, float]:
        r"""Calculate :math:`\\langle \\sigma v \rangle` for a given ion temperature.

        Cross-section from :cite:`bosch_improved_1992`

        Args:
            ion_temp: [keV]

        Returns:
            :math:`\\langle \\sigma v \rangle` [cm^3/s]
        """
        # For D(d,n)3He
        cBH_1 = [((31.3970**2) / 4.0) ** (1.0 / 3.0), 5.65718e-12, 3.41e-03, 1.99e-03, 0, 1.05e-05, 0, 0]  # 3.72e-16,

        mc2_1 = 937814.0

        # For D(d,p)T
        cBH_2 = [((31.3970**2) / 4.0) ** (1.0 / 3.0), 5.43360e-12, 5.86e-03, 7.68e-03, 0, -2.96e-06, 0, 0]  # 3.57e-16,

        mc2_2 = 937814.0

        thetaBH_1 = ion_temp / (
            1
            - (
                (cBH_1[2] * ion_temp + cBH_1[4] * ion_temp**2 + cBH_1[6] * ion_temp**3)
                / (1 + cBH_1[3] * ion_temp + cBH_1[5] * ion_temp**2 + cBH_1[7] * ion_temp**3)
            )
        )

        thetaBH_2 = ion_temp / (
            1
            - (
                (cBH_2[2] * ion_temp + cBH_2[4] * ion_temp**2 + cBH_2[6] * ion_temp**3)
                / (1 + cBH_2[3] * ion_temp + cBH_2[5] * ion_temp**2 + cBH_2[7] * ion_temp**3)
            )
        )

        etaBH_1 = cBH_1[0] / (thetaBH_1 ** (1.0 / 3.0))
        etaBH_2 = cBH_2[0] / (thetaBH_2 ** (1.0 / 3.0))

        sigmav_DD_to_nHe3 = cBH_1[1] * thetaBH_1 * np.sqrt(etaBH_1 / (mc2_1 * (ion_temp**3.0))) * np.exp(-3.0 * etaBH_1)
        sigmav_DD_to_pT = cBH_2[1] * thetaBH_2 * np.sqrt(etaBH_2 / (mc2_2 * (ion_temp**3.0))) * np.exp(-3.0 * etaBH_2)

        sigmav_combined = sigmav_DD_to_pT + sigmav_DD_to_nHe3
        if not np.isreal(sigmav_combined):
            return np.nan, np.nan, np.nan
        else:
            return sigmav_combined, sigmav_DD_to_pT, sigmav_DD_to_nHe3

    def calc_average_fuel_ion_mass(self) -> Unitfull:
        """Returns the average mass of the fuel ions.

        Returns:
            :term:`fuel_average_mass_number` [amu]
        """
        return 2.0 * ureg.amu

    def calc_energy_per_reaction(self, ion_temp: Unitfull) -> Unitfull:
        """Returns the total energy per reaction."""
        sigmav_combined, sigmav_DD_to_pT, sigmav_DD_to_nHe3 = self.calc_rate_coefficient(ion_temp)

        energy_per_reaction = (
            self.energy_per_DD_to_pT_reaction * sigmav_DD_to_pT + self.energy_per_DD_to_nHe3_reaction * sigmav_DD_to_nHe3
        ) / sigmav_combined

        return convert_units(energy_per_reaction, ureg.MJ)

    def calc_energy_to_neutrals_per_reaction(self, ion_temp: Unitfull) -> Unitfull:
        """Returns the energy going to uncharged species per reaction."""
        sigmav_combined, sigmav_DD_to_pT, sigmav_DD_to_nHe3 = self.calc_rate_coefficient(ion_temp)

        energy_to_neutrals_per_reaction = (
            self.energy_to_neutrals_per_DD_to_pT_reaction * sigmav_DD_to_pT
            + self.energy_to_neutrals_per_DD_to_nHe3_reaction * sigmav_DD_to_nHe3
        ) / sigmav_combined

        return convert_units(energy_to_neutrals_per_reaction, ureg.MJ)

    def calc_energy_to_charged_per_reaction(self, ion_temp: Unitfull) -> Unitfull:
        """Returns the energy going to charged species per reaction."""
        sigmav_combined, sigmav_DD_to_pT, sigmav_DD_to_nHe3 = self.calc_rate_coefficient(ion_temp)

        energy_to_charged_per_reaction = (
            self.energy_to_charged_per_DD_to_pT_reaction * sigmav_DD_to_pT
            + self.energy_to_charged_per_DD_to_nHe3_reaction * sigmav_DD_to_nHe3
        ) / sigmav_combined

        return convert_units(energy_to_charged_per_reaction, ureg.MJ)

    def calc_power_density(self, ion_temp: Unitfull) -> Unitfull:
        """Returns the total power density."""
        _, sigmav_DD_to_pT, sigmav_DD_to_nHe3 = self.calc_rate_coefficient(ion_temp)

        power_density = sigmav_DD_to_pT * self.energy_per_DD_to_pT_reaction + sigmav_DD_to_nHe3 * self.energy_per_DD_to_nHe3_reaction

        return convert_units(power_density, ureg.MW * ureg.m**3)

    def calc_power_density_to_neutrals(self, ion_temp: Unitfull) -> Unitfull:
        """Returns the power density going to uncharged species."""
        _, sigmav_DD_to_pT, sigmav_DD_to_nHe3 = self.calc_rate_coefficient(ion_temp)

        power_density = (
            sigmav_DD_to_pT * self.energy_to_neutrals_per_DD_to_pT_reaction
            + sigmav_DD_to_nHe3 * self.energy_to_neutrals_per_DD_to_nHe3_reaction
        )

        return convert_units(power_density, ureg.MW * ureg.m**3)

    def calc_power_density_to_charged(self, ion_temp: Unitfull) -> Unitfull:
        """Returns the power density going to charged species."""
        _, sigmav_DD_to_pT, sigmav_DD_to_nHe3 = self.calc_rate_coefficient(ion_temp)

        power_density = (
            sigmav_DD_to_pT * self.energy_to_charged_per_DD_to_pT_reaction
            + sigmav_DD_to_nHe3 * self.energy_to_charged_per_DD_to_nHe3_reaction
        )

        return convert_units(power_density, ureg.MW * ureg.m**3)


class DDFusionHively(DDFusionBoschHale):
    """Deuterium-Deuterium reaction using Hively cross-section."""

    @staticmethod
    @wraps_ufunc(
        input_units=dict(ion_temp=ureg.keV),
        return_units=dict(sigmav_combined=ureg.cm**3 / ureg.s, sigmav_DD_to_pT=ureg.cm**3 / ureg.s, sigmav_DD_to_nHe3=ureg.cm**3 / ureg.s),
        output_core_dims=[(), (), ()],
    )
    def calc_rate_coefficient(ion_temp: float) -> tuple[float, float, float]:
        r"""Calculate :math:`\\langle \\sigma v \rangle` for a given ion temperature.

        Cross-section from :cite:`hively_convenient_1977`

        Args:
            ion_temp: [keV]

        Returns:
            :math:`\\langle \\sigma v \rangle` [cm^3/s]
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
        sigmav_DD_to_pT = np.exp(
            a_1[0] / ion_temp**r_1 + a_1[1] + a_1[2] * ion_temp + a_1[3] * ion_temp**2.0 + a_1[4] * ion_temp**3.0 + a_1[5] * ion_temp**4.0
        )
        sigmav_DD_to_nHe3 = np.exp(
            a_2[0] / ion_temp**r_2 + a_2[1] + a_2[2] * ion_temp + a_2[3] * ion_temp**2.0 + a_2[4] * ion_temp**3.0 + a_2[5] * ion_temp**4.0
        )
        sigmav_combined = sigmav_DD_to_pT + sigmav_DD_to_nHe3
        return sigmav_combined, sigmav_DD_to_pT, sigmav_DD_to_nHe3


class DHe3Fusion(FusionReaction):
    """Deuterium-Helium-3 reaction (Bosch-Hale cross-section)."""

    def __init__(self) -> None:
        """Sets the reaction energies for the Deuterium-Tritium reaction."""
        super().__init__()
        self.energy_per_reaction = Quantity(18.3, ureg.MeV)
        self.energy_to_neutrals_per_reaction = Quantity(0.0, ureg.MeV)
        self.energy_to_charged_per_reaction = self.energy_per_reaction

    @staticmethod
    @wraps_ufunc(input_units=dict(ion_temp=ureg.keV), return_units=dict(sigmav=ureg.cm**3 / ureg.s))
    def calc_rate_coefficient(ion_temp: float) -> float:
        r"""Deuterium-Helium-3 reaction.

        Calculate :math:`\langle \sigma v \rangle` for a given characteristic ion energy.

        Function tested on available data at [1, 2, 5, 10, 20, 50, 100] keV.
        Maximum error = 8.4% within range 2-100 keV and should not be used outside range [2, 100] keV.

        Uses DD cross section formulation :cite:`bosch_improved_1992`.

        Args:
            ion_temp: [keV]

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

        thetaBH_1 = ion_temp / (
            1
            - (
                (cBH_1[2] * ion_temp + cBH_1[4] * ion_temp**2 + cBH_1[6] * ion_temp**3)
                / (1 + cBH_1[3] * ion_temp + cBH_1[5] * ion_temp**2 + cBH_1[7] * ion_temp**3.0)
            )
        )

        etaBH_1 = cBH_1[0] / (thetaBH_1 ** (1.0 / 3.0))

        sigmav = cBH_1[1] * thetaBH_1 * np.sqrt(etaBH_1 / (mc2_1 * (ion_temp**3.0))) * np.exp(-3.0 * etaBH_1)

        return float(sigmav)

    def calc_average_fuel_ion_mass(self, heavier_fuel_species_fraction: Unitfull) -> Unitfull:
        """Calculate the average mass of the fuel ions.

        Args:
            heavier_fuel_species_fraction: n_heavier / (n_heavier + n_lighter) number fraction.

        Returns:
            :term:`fuel_average_mass_number` [amu]
        """
        average_fuel_ion_mass = 2.0 * (1 - heavier_fuel_species_fraction) + 3.0 * heavier_fuel_species_fraction
        return average_fuel_ion_mass * ureg.amu

    def calc_energy_per_reaction(self) -> Unitfull:
        """Returns the total energy per reaction."""
        return convert_units(self.energy_per_reaction, ureg.MJ)

    def calc_energy_to_neutrals_per_reaction(self) -> Unitfull:
        """Returns the energy going to uncharged species per reaction."""
        return convert_units(self.energy_to_neutrals_per_reaction, ureg.MJ)

    def calc_energy_to_charged_per_reaction(self) -> Unitfull:
        """Returns the energy going to charged species per reaction."""
        return convert_units(self.energy_to_charged_per_reaction, ureg.MJ)

    def calc_power_density(self, ion_temp: Unitfull, heavier_fuel_species_fraction: Unitfull) -> Unitfull:
        """Returns the total power density."""
        sigmav = self.calc_rate_coefficient(ion_temp)
        fuel_ratio = heavier_fuel_species_fraction * (1.0 - heavier_fuel_species_fraction)

        power_density = sigmav * self.energy_per_reaction * fuel_ratio

        return convert_units(power_density, ureg.MW * ureg.m**3)

    def calc_power_density_to_neutrals(self) -> Unitfull:
        """Returns the power density going to uncharged species."""
        return Quantity(0.0, ureg.MW * ureg.m**3)

    def calc_power_density_to_charged(self, ion_temp: Unitfull, heavier_fuel_species_fraction: Unitfull) -> Unitfull:
        """Returns the power density going to charged species."""
        return self.calc_power_density(ion_temp, heavier_fuel_species_fraction)


class pB11Fusion(FusionReaction):
    """Proton-Boron-11 reaction (Nevins-Swain cross-section)."""

    def __init__(self) -> None:
        """Sets the reaction energies for the Deuterium-Tritium reaction."""
        super().__init__()
        self.energy_per_reaction = Quantity(8.7, ureg.MeV)
        self.energy_to_neutrals_per_reaction = Quantity(0.0, ureg.MeV)
        self.energy_to_charged_per_reaction = self.energy_per_reaction

    @staticmethod
    @wraps_ufunc(input_units=dict(ion_temp=ureg.keV), return_units=dict(sigmav=ureg.cm**3 / ureg.s))
    def calc_rate_coefficient(ion_temp: float) -> float:
        r"""Proton (hydrogen)-Boron11 reaction.

        Calculate :math:`\langle \sigma v \rangle` for a given characteristic ion energy.

        Uses cross section from Nevins and Swain :cite:`nevins_thermonuclear_2000`.
        Updated cross sections in :cite:`sikora_new_2016`, and :cite:`putvinski_fusion_2019` are not in analytic form.

        Args:
            ion_temp: [keV]

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

        thetaBH_1 = ion_temp / (
            1
            - (
                (cBH_1[2] * ion_temp + cBH_1[4] * ion_temp**2 + cBH_1[6] * ion_temp**3)
                / (1 + cBH_1[3] * ion_temp + cBH_1[5] * ion_temp**2 + cBH_1[7] * ion_temp**3)
            )
        )

        etaBH_1 = cBH_1[0] / (thetaBH_1 ** (1.0 / 3.0))

        sigmavNRhigh = (cBH_1[1] * thetaBH_1 * np.sqrt(etaBH_1 / (mc2_1 * (ion_temp**3.0))) * np.exp(-3.0 * etaBH_1)) * 1e6  # m3 to cm3

        # Low temperature (T<60 keV)
        E0 = ((17.81) ** (1.0 / 3.0)) * (ion_temp ** (2.0 / 3.0))

        deltaE0 = 4.0 * np.sqrt(ion_temp * E0 / 3.0)

        tau = (3.0 * E0) / ion_temp

        Mp = 1.0  # *1.67e-27
        MB = 11.0  # *1.67e-27

        Mr = (Mp * MB) / (Mp + MB)

        C0 = 197.000 * 1e-25  # MeVb to kev/m^2
        C1 = 0.240 * 1e-25  # MeVb to kev/m^2
        C2 = 0.000231 * 1e-25  # MeVb to kev/m^2

        Seff = C0 * (1 + (5.0 / (12.0 * tau))) + C1 * (E0 + (35.0 / 36.0) * ion_temp) + C2 * (E0**2.0 + (89.0 / 36.0) * E0 * ion_temp)

        sigmavNRlow = (np.sqrt(2 * ion_temp / Mr) * ((deltaE0 * Seff) / (ion_temp ** (2.0))) * np.exp(-tau)) * 1e6  # m3 to cm3
        # 148 keV resonance
        sigmavR = ((5.41e-21) * ((1.0 / ion_temp) ** (3.0 / 2.0)) * np.exp(-148.0 / ion_temp)) * 1e6  # m3 to cm3
        sigmav = sigmavNRhigh

        if ion_temp < 60.0:  # keV
            sigmav = sigmavNRlow + sigmavR
        elif (ion_temp > 60.0) and (ion_temp < 130):  # keV
            sigmav = sigmavNRhigh + sigmavR

        return float(sigmav)

    def calc_average_fuel_ion_mass(self, heavier_fuel_species_fraction: Unitfull) -> Unitfull:
        """Calculate the average mass of the fuel ions.

        Args:
            heavier_fuel_species_fraction: n_heavier / (n_heavier + n_lighter) number fraction.

        Returns:
            :term:`fuel_average_mass_number` [amu]
        """
        average_fuel_ion_mass = 1.0 * (1 - heavier_fuel_species_fraction) + 11.0 * heavier_fuel_species_fraction
        return average_fuel_ion_mass * ureg.amu

    def calc_energy_per_reaction(self) -> Unitfull:
        """Returns the total energy per reaction."""
        return convert_units(self.energy_per_reaction, ureg.MJ)

    def calc_energy_to_neutrals_per_reaction(self) -> Unitfull:
        """Returns the energy going to uncharged species per reaction."""
        return convert_units(self.energy_to_neutrals_per_reaction, ureg.MJ)

    def calc_energy_to_charged_per_reaction(self) -> Unitfull:
        """Returns the energy going to charged species per reaction."""
        return convert_units(self.energy_to_charged_per_reaction, ureg.MJ)

    def calc_power_density(self, ion_temp: Unitfull, heavier_fuel_species_fraction: Unitfull) -> Unitfull:
        """Returns the total power density."""
        sigmav = self.calc_rate_coefficient(ion_temp)
        fuel_ratio = heavier_fuel_species_fraction * (1.0 - heavier_fuel_species_fraction)

        power_density = sigmav * self.energy_per_reaction * fuel_ratio

        return convert_units(power_density, ureg.MW * ureg.m**3)

    def calc_power_density_to_neutrals(self) -> Unitfull:
        """Returns the power density going to uncharged species."""
        return Quantity(0.0, ureg.MW * ureg.m**3)

    def calc_power_density_to_charged(self, ion_temp: Unitfull, heavier_fuel_species_fraction: Unitfull) -> Unitfull:
        """Returns the power density going to charged species."""
        return self.calc_power_density(ion_temp, heavier_fuel_species_fraction)


REACTIONS = dict(
    DTFusionBoschHale=DTFusionBoschHale(),
    DTFusionHively=DTFusionHively(),
    DDFusionBoschHale=DDFusionBoschHale(),
    DDFusionHively=DDFusionHively(),
    DHe3Fusion=DHe3Fusion(),
    pB11Fusion=pB11Fusion(),
)
