from __future__ import annotations

from typing import Any, ClassVar
from abc import ABC, abstractmethod
from cfspopcon.unit_handling import Unitfull, convert_units, ureg, Quantity, wraps_ufunc
import numpy as np

class FusionReaction(ABC):

    instances: ClassVar[dict[str, FusionReaction]] = dict()

    def __init__(self) -> None:
        self.instances[self.__class__.__name__] = self

    @abstractmethod
    def calc_average_fuel_ion_mass(heavier_fuel_species_fraction: Unitfull):
        pass

    @abstractmethod
    def calc_energy_per_reaction(self):
        pass
    
    @abstractmethod
    def calc_energy_to_neutrals_per_reaction(self):
        pass

    @abstractmethod
    def calc_energy_to_charged_per_reaction(self):
        pass

    @abstractmethod
    def calc_power_density(self):
        pass
    
    @abstractmethod
    def calc_power_density_to_neutrals(self):
        pass

    @abstractmethod
    def calc_power_density_to_charged(self):
        pass

class DTFusionBoschHale(FusionReaction):

    def __init__(self):
        self.energy_per_reaction = Quantity(17.6, ureg.MeV)
        self.energy_to_neutrals_per_reaction = self.energy_per_reaction * 4.0 / 5.0
        self.energy_to_charged_per_reaction = self.energy_per_reaction * 1.0 / 5.0
    
    @staticmethod
    @wraps_ufunc(
        input_units=dict(ion_temp=ureg.keV),
        return_units=dict(sigmav=ureg.cm**3 / ureg.s)
    )
    def calc_rate_coefficient(ion_temp):
        C = [0.0, 1.173e-9, 1.514e-2, 7.519e-2, 4.606e-3, 1.35e-2, -1.068e-4, 1.366e-5]
        B_G = 34.3827
        mr_c2 = 1124656

        theta = ion_temp / (
            1
            - (ion_temp * (C[2] + ion_temp * (C[4] + ion_temp * C[6])))
            / (1 + ion_temp * (C[3] + ion_temp * (C[5] + ion_temp * C[7])))
        )
        eta = (B_G**2 / (4 * theta)) ** (1 / 3)
        sigmav = C[1] * theta * np.sqrt(eta / (mr_c2 * ion_temp**3)) * np.exp(-3 * eta)

        return sigmav

    @staticmethod
    def calc_average_fuel_ion_mass(heavier_fuel_species_fraction: Unitfull):
        average_fuel_ion_mass = 2.0 * (1 - heavier_fuel_species_fraction) + 3.0 * heavier_fuel_species_fraction
        return average_fuel_ion_mass * ureg.amu
    
    def calc_energy_per_reaction(self):
        return convert_units(self.energy_per_reaction, ureg.MJ)
    
    def calc_energy_to_neutrals_per_reaction(self):
        return convert_units(self.energy_to_neutrals_per_reaction, ureg.MJ)
    
    def calc_energy_to_charged_per_reaction(self):
        return convert_units(self.energy_to_charged_per_reaction, ureg.MJ)
    
    def calc_power_density(self, ion_temp: Unitfull, heavier_fuel_species_fraction: Unitfull):
        sigmav = self.calc_rate_coefficient(ion_temp)
        fuel_ratio = heavier_fuel_species_fraction * (1.0 - heavier_fuel_species_fraction)

        power_density = (
            sigmav * self.energy_per_reaction * fuel_ratio
        )

        return convert_units(power_density, ureg.MW * ureg.m**3)

    def calc_power_density_to_neutrals(self, ion_temp: Unitfull, heavier_fuel_species_fraction: Unitfull):
        return (4.0 / 5.0) * self.calc_power_density(ion_temp, heavier_fuel_species_fraction)

    def calc_power_density_to_charged(self, ion_temp: Unitfull, heavier_fuel_species_fraction: Unitfull):
        return (1.0 / 5.0) * self.calc_power_density(ion_temp, heavier_fuel_species_fraction)

class DTFusionHively(DTFusionBoschHale):

    @staticmethod
    @wraps_ufunc(
        input_units=dict(ion_temp=ureg.keV),
        return_units=dict(sigmav=ureg.cm**3 / ureg.s)
    )
    def calc_rate_coefficient(ion_temp):
        A = [-21.377692, -25.204054, -7.1013427 * 1e-2, 1.9375451 * 1e-4, 4.9246592 * 1e-6, -3.9836572 * 1e-8]
        r = 0.2935
        sigmav = np.exp(
            A[0] / ion_temp**r
            + A[1]
            + A[2] * ion_temp
            + A[3] * ion_temp**2.0
            + A[4] * ion_temp**3.0
            + A[5] * ion_temp**4.0
        )
        return sigmav
