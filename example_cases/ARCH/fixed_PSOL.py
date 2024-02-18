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

print(dataset['P_sol'].values)

target = 50
epsilon = 1e-5
delta_imp = 1e-9
initial_impurity = [0.0]

impurities_array = np.zeros((dataset['average_electron_density'].size, dataset['average_electron_temp'].size))
P_sols_array = np.zeros((dataset['average_electron_density'].size, dataset['average_electron_temp'].size))

for i in range(impurities_array.shape[0]): # some elements have P_sol = 0, should we not count them?
    for j in range(impurities_array.shape[1]): # some elements have P_sol = 0, should we not count them?
        
        print(f"(i, j) = ({i}, {j})")

        dataset['impurities'].values = ureg.Quantity(initial_impurity, 'dimensionless') # Set initial impurity to initial_impurity = [0.0]
        algorithm.update_dataset(dataset, in_place=True)

        if np.abs(dataset['P_sol'].values[i,j]) < target:
            impurities_array[i,j] = initial_impurity[0]
            continue
        while(np.abs(dataset['P_sol'].values[i][j] - target) > epsilon):

            # Approximate partial derivative of P_sol WRT impurity concentration (at an impurity concentration imp_0)

            imp_0 = dataset['impurities'].values
            P_sol_0 = dataset['P_sol'].values[i,j]

            print(imp_0, P_sol_0)
            
            imp_next = imp_0 + delta_imp
            dataset['impurities'].values = ureg.Quantity(imp_next, 'dimensionless')
            algorithm.update_dataset(dataset, in_place=True)
            P_sol_next = dataset['P_sol'].values[i,j]

            derivative = (P_sol_next - P_sol_0) / delta_imp

            """imp_prev = imp_0 - delta_imp
            dataset['impurities'].values = ureg.Quantity(imp_prev, 'dimensionless')
            algorithm.update_dataset(dataset, in_place=True)
            P_sol_prev = dataset['P_sol'].values[i,j]

            derivative2 = P_sol_next - 2*P_sol_0 + P_sol_prev / delta_imp**2 """

            if(derivative == 0):
                break

            # Adjust impurity concentration and recalculate other parameters
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
            

print(P_sols_array, "\n")
print(impurities_array)

# If you run this code, you will notice that there is a negative impurity concentration for impurities_array[1, 2]...what?
# This seems wrong, but looking at the graphs in P_sol_vs_impurities.png, this makes sense. 
# There is actually no impurity concentration for which dataset['P_sol'].values[1, 2] = 50.
# The negative value is an artifact of using Newton's algorithm.

