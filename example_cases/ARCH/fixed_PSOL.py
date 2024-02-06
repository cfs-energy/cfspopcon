import xarray as xr
import numpy as np
import os
import cfspopcon
from cfspopcon.unit_handling import ureg
import warnings
from pint import UnitStrippedWarning

warnings.filterwarnings("ignore", category=UnitStrippedWarning)

# read in .yaml file
input_file_path = os.path.join(os.getcwd(), "example_cases/ARCH/input_arch.yaml")
input_parameters, algorithm, points = cfspopcon.read_case(input_file_path)
algorithm.validate_inputs(input_parameters)
dataset = xr.Dataset(input_parameters)
algorithm.update_dataset(dataset, in_place=True)

# For each i index in the density array and j index in the temp array, 
# change impurity concentration and re-compute all quantities.
# Then, add (i,j) to the "data" array.
# This helps us obtain the density and temperature at which P_sol = target.

target = 50
impurities_array = np.zeros((dataset['average_electron_density'].size, dataset['average_electron_temp'].size))

for i in range(0, impurities_array.shape[0]): # some elements have P_sol = 0, should we not count them?
    for j in range(0, impurities_array.shape[1]): # some elements have P_sol = 0, should we not count them?
        diff = np.abs(dataset['P_sol'].values[i,j] - target)
        while(np.abs(diff) > 1.0):
            increment = np.sign(diff) * 1e-3 # Find a better formula for increment!! Very slow 😭
            dataset['impurities'].values = ureg.Quantity(dataset['impurities'].values + increment, 'dimensionless')
            algorithm.update_dataset(dataset, in_place=True) # Getting invalid arguments for logarithms for some reason.
            print(dataset['P_sol'].values[i,j])
            impurities_array[i,j] = dataset['P_sol'].values[i,j]
            diff = np.abs(dataset['P_sol'].values[i,j] - target)