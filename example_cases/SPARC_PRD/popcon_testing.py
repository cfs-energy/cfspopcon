import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cfspopcon
from cfspopcon.unit_handling import ureg

input_parameters, algorithm, points, plots  = cfspopcon.read_case("./example_cases/SPARC_PRD/input.yaml")
algorithm.validate_inputs(input_parameters, raise_error_on_missing_inputs=False)
dataset = xr.Dataset(input_parameters)
dataset = algorithm.update_dataset(dataset)

P_sol = dataset["P_sol"].to_numpy()
P_sol_target = dataset["P_sol"].to_numpy()
np.set_printoptions(threshold=np.Inf)

plot_style = cfspopcon.read_plot_style("./example_cases/SPARC_PRD/plot_popcon.yaml")
cfspopcon.plotting.make_plot(
    dataset,
    plot_style,
    points=points,
    title=f"Scaling: {dataset['tau_e_scaling'].values}",
    output_dir=None
)
plt.show()