import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cfspopcon
from cfspopcon.unit_handling import ureg

input_parameters, algorithm, points, plots  = cfspopcon.read_case("../../example_cases/SPARC_PRD/input.yaml")
algorithm.validate_inputs(input_parameters)
dataset = xr.Dataset(input_parameters)
algorithm.update_dataset(dataset)

print()
P_sol = dataset["P_sol"].to_numpy()
P_sol_target = dataset["P_sol"].to_numpy()
np.set_printoptions(threshold=np.Inf)
print(P_sol)


plot_style = cfspopcon.read_plot_style("../../example_cases/SPARC_PRD/plot_P_sol_only.yaml")
cfspopcon.plotting.make_plot(
    dataset,
    plot_style,
    points=points,
    title="POPCON example",
    output_dir=None
)
plt.show()
