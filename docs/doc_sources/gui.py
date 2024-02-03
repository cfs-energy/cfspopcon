import base64
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import panel as pn

pn.extension(design="material")
import xarray as xr

import cfspopcon
from cfspopcon.unit_handling import ureg

if __name__ == "__main__":
    input_parameters, algorithm, points = cfspopcon.read_case("../../example_cases/SPARC_PRD")
    plot_style = cfspopcon.read_plot_style("../../example_cases/SPARC_PRD/plot_popcon.yaml")
    _, ax = plt.subplots(figsize=plot_style["figsize"], dpi=plot_style["show_dpi"])
    algorithm.validate_inputs(input_parameters)
    dataset = xr.Dataset(input_parameters)

    def make_plot_for_B(magnetic_field_on_axis: float, ax):
        dataset["magnetic_field_on_axis"] = magnetic_field_on_axis * ureg("T")
        algorithm.update_dataset(dataset, in_place=True)
        fig, ax = cfspopcon.plotting.make_plot(dataset, plot_style, points=points, title="POPCON example", output_dir=None, ax=ax)
        return fig, ax

    def plot_figure(magnetic_field_on_axis, ax=ax):
        fig, ax = make_plot_for_B(magnetic_field_on_axis, ax)
        return pn.pane.Matplotlib(fig)

    slider = pn.widgets.FloatSlider(name="Magnetic Field on Axis", start=5.0, end=15.0, step=0.1, value=12.2)
    interactive_plot = pn.bind(plot_figure, magnetic_field_on_axis=slider)
    app = pn.Column(slider, interactive_plot)
    app.servable()
