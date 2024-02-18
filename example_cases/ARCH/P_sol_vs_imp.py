import xarray as xr
import numpy as np
import os
import cfspopcon
from cfspopcon.unit_handling import ureg
import warnings
from pint import UnitStrippedWarning
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UnitStrippedWarning)

# read in .yaml file
input_file_path = os.path.join(os.getcwd(), "example_cases/ARCH/input_arch.yaml")
input_parameters, algorithm, points = cfspopcon.read_case(input_file_path)
algorithm.validate_inputs(input_parameters)
dataset = xr.Dataset(input_parameters)
algorithm.update_dataset(dataset, in_place=True)

n = 300 # number of points to plot

fig, axs = plt.subplots(1, 1)
plt.subplots_adjust(hspace=0.1)

imps = np.linspace(0.0, 0.0005, n)

for i in range(1, 2):
    for j in range(3, 4):
        P_sols = np.zeros(n)
        for index in range(imps.size):
            print(i, j, index)
            dataset['impurities'].values = ureg.Quantity([imps[index]], 'dimensionless')
            algorithm.update_dataset(dataset, in_place=True)
            P_sols[index] = dataset['P_sol'].values[i,j]
            print(imps[index], P_sols[index])
        axs.plot(imps, P_sols)
plt.show()
