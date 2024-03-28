import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import cfspopcon
from cfspopcon.unit_handling import ureg

print("HELLO WORLD")
input_parameters, algorithm, points  = cfspopcon.read_case("../../example_cases/SPARC_PRD/input.yaml")
algorithm.validate_inputs(input_parameters)
dataset = xr.Dataset(input_parameters)
algorithm.update_dataset(dataset, in_place=True)

print(dataset["minimum_core_radiated_fraction"])
