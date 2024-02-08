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
        print("INDICES", i, j)
        while(np.abs(dataset['P_sol'].values[i][j] - target) > 0.5):
            
            # Approximate partial derivative of P_sol WRT impurity concentration (at an impurity concentration imp_0)
            imp_0 = dataset['impurities'].values
            P_sol_0 = dataset['P_sol'].values[i][j]
            imp_1 = imp_0 + 1e-7
            dataset['impurities'].values = ureg.Quantity(imp_1, 'dimensionless')
            algorithm.update_dataset(dataset, in_place=True)
            P_sol_1 = dataset['P_sol'].values[i][j]
            derivative = (P_sol_1 - P_sol_0) / (imp_1 - imp_0)

            if(derivative == 0):
                break

            # Adjust impurity concentration and recalculate other parameters
            imp_0 = imp_0 - (P_sol_0 - target) / derivative
            dataset['impurities'].values = ureg.Quantity(imp_0, 'dimensionless')
            algorithm.update_dataset(dataset, in_place=True)

            print(f"impurity: {imp_0}...P_sol: {dataset['P_sol'].values[i][j]}")