"""Calculate the power radiated from the confined region due to the fuel and impurity species."""
import xarray as xr

from .. import formulas, named_options
from ..atomic_data import read_atomic_data
from ..unit_handling import Unitfull, convert_to_default_units, ureg
from .algorithm_class import Algorithm

import xarray as xr
import numpy as np
import os
import cfspopcon
import warnings
from pint import UnitStrippedWarning

warnings.filterwarnings("ignore", category=UnitStrippedWarning)

RETURN_KEYS = ["P_radiation", "impurities_for_given_P_SOL"]


def run_calc_core_radiated_power(
    rho: Unitfull,
    electron_density_profile: Unitfull,
    electron_temp_profile: Unitfull,
    z_effective: Unitfull,
    plasma_volume: Unitfull,
    major_radius: Unitfull,
    minor_radius: Unitfull,
    magnetic_field_on_axis: Unitfull,
    separatrix_elongation: Unitfull,
    radiated_power_method: named_options.RadiationMethod,
    radiated_power_scalar: Unitfull,
    impurities: xr.DataArray,
) -> dict[str, Unitfull]:
    """Calculate the power radiated from the confined region due to the fuel and impurity species.

    Args:
        rho: :term:`glossary link<rho>`
        electron_density_profile: :term:`glossary link<electron_density_profile>`
        electron_temp_profile: :term:`glossary link<electron_temp_profile>`
        z_effective: :term:`glossary link<z_effective>`
        plasma_volume: :term:`glossary link<plasma_volume>`
        major_radius: :term:`glossary link<major_radius>`
        minor_radius: :term:`glossary link<minor_radius>`
        magnetic_field_on_axis: :term:`glossary link<magnetic_field_on_axis>`
        separatrix_elongation: :term:`glossary link<separatrix_elongation>`
        radiated_power_method: :term:`glossary link<radiated_power_method>`
        radiated_power_scalar: :term:`glossary link<radiated_power_scalar>`
        impurities: :term:`glossary link<impurities>`

    Returns:
        :term:`P_radiation`

    """
    P_rad_bremsstrahlung = formulas.calc_bremsstrahlung_radiation(
        rho, electron_density_profile, electron_temp_profile, z_effective, plasma_volume
    )
    P_rad_bremsstrahlung_from_hydrogen = formulas.calc_bremsstrahlung_radiation(
        rho, electron_density_profile, electron_temp_profile, 1.0, plasma_volume
    )
    P_rad_synchrotron = formulas.calc_synchrotron_radiation(
        rho,
        electron_density_profile,
        electron_temp_profile,
        major_radius,
        minor_radius,
        magnetic_field_on_axis,
        separatrix_elongation,
        plasma_volume,
    )

    # Calculate radiated power due to Bremsstrahlung, Synchrotron and impurities
    if radiated_power_method == named_options.RadiationMethod.Inherent:
        P_radiation = radiated_power_scalar * (P_rad_bremsstrahlung + P_rad_synchrotron)
    else:
        atomic_data = read_atomic_data()

        P_rad_impurity = formulas.calc_impurity_radiated_power(
            radiated_power_method=radiated_power_method,
            rho=rho,
            electron_temp_profile=electron_temp_profile,
            electron_density_profile=electron_density_profile,
            impurities=impurities,
            plasma_volume=plasma_volume,
            atomic_data=atomic_data,
        )

        P_radiation = radiated_power_scalar * (
            P_rad_bremsstrahlung_from_hydrogen + P_rad_synchrotron + P_rad_impurity.sum(dim="dim_species")
        )

    local_vars = locals()
    return {key: convert_to_default_units(local_vars[key], key) for key in RETURN_KEYS}


def calc_impurities_for_given_P_SOL(constant_P_SOL):
    input_file_path = os.path.join(os.getcwd(), "example_cases/ARCH/input_arch.yaml")
    input_parameters, algorithm, points = cfspopcon.read_case(input_file_path)
    algorithm.validate_inputs(input_parameters)
    dataset = xr.Dataset(input_parameters)
    algorithm.update_dataset(dataset, in_place=True)

    target = 50
    epsilon = 1e-5
    delta_imp = 1e-9
    initial_impurity = [0.0]

    impurities_array = np.zeros((dataset['average_electron_density'].size, dataset['average_electron_temp'].size))
    P_sols_array = np.zeros((dataset['average_electron_density'].size, dataset['average_electron_temp'].size))

    for i in range(impurities_array.shape[0]): # some elements have P_sol = 0, should we not count them?
        for j in range(impurities_array.shape[1]): # some elements have P_sol = 0, should we not count them?
            dataset['impurities'].values = ureg.Quantity(initial_impurity, 'dimensionless') # Set initial impurity to initial_impurity = [0.0]
            algorithm.update_dataset(dataset, in_place=True)

            if np.abs(dataset['P_sol'].values[i,j]) < target:
                impurities_array[i,j] = initial_impurity[0]
                continue
            while(np.abs(dataset['P_sol'].values[i][j] - target) > epsilon):
                imp_0 = dataset['impurities'].values
                P_sol_0 = dataset['P_sol'].values[i,j]
                imp_next = imp_0 + delta_imp
                dataset['impurities'].values = ureg.Quantity(imp_next, 'dimensionless')
                algorithm.update_dataset(dataset, in_place=True)
                P_sol_next = dataset['P_sol'].values[i,j]
                derivative = (P_sol_next - P_sol_0) / delta_imp
                if(derivative == 0):
                    break

                imp_0 = imp_0 - (P_sol_0 - target) / derivative
                dataset['impurities'].values = ureg.Quantity(imp_0, 'dimensionless')
                algorithm.update_dataset(dataset, in_place=True)

                while(dataset['P_sol'].values[i,j] == 0.0):
                    print(imp_0, dataset['P_sol'].values[i][j])
                    imp_0 = imp_0 * 0.99
                    dataset['impurities'].values = ureg.Quantity(imp_0, 'dimensionless')
                    algorithm.update_dataset(dataset, in_place=True)
                
                impurities_array[i,j] = imp_0
                P_sols_array[i,j] = dataset['P_sol'].values[i,j]
    return impurities_array

impurities_for_given_P_SOL = Algorithm(
    function=calc_impurities_for_given_P_SOL,
    return_keys=RETURN_KEYS
)

calc_core_radiated_power = Algorithm(
    function=run_calc_core_radiated_power,
    return_keys=RETURN_KEYS,
)
