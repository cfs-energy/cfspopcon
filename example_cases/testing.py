import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import cfspopcon
from cfspopcon.unit_handling import ureg

input_parameters, algorithm, points = cfspopcon.read_case("./SPARC_PRD/input.yaml")
algorithm.validate_inputs(input_parameters)
dataset = xr.Dataset(input_parameters)
dataset = algorithm.update_dataset(dataset)

print("hello world")
plot_style = cfspopcon.read_plot_style("./SPARC_PRD/plot_popcon.yaml")
cfspopcon.plotting.make_plot(dataset, plot_style, points=points, title="POPCON example", output_dir=None)
plt.show()