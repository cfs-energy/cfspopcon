"""Plot creation functions."""

from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.contour
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes

from ..shaping_and_selection.point_selection import build_mask_from_dict, find_values_at_nearest_point
from ..shaping_and_selection.transform_coords import build_transform_function_from_dict
from ..unit_handling import Quantity, Unit, default_unit, dimensionless_magnitude, magnitude_in_units
from .coordinate_formatter import CoordinateFormatter


def make_plot(
    dataset: xr.Dataset,
    plot_params: dict,
    points: dict,
    title: str,
    save_name: Optional[str] = None,
    ax: Optional[Axes] = None,
    output_dir: Path = Path("."),
):
    """Given a dictionary corresponding to a plotting style, build a standard plot from the results of the POPCON."""
    if plot_params["type"] == "popcon":
        if ax is None:
            _, ax = plt.subplots(figsize=plot_params["figsize"], dpi=plot_params["show_dpi"])
        fig, ax = make_popcon_plot(dataset, title, plot_params, points, ax=ax)
    else:
        raise NotImplementedError(f"No plotting method for type '{plot_params['type']}'")

    if save_name is not None:
        fig.savefig(output_dir / save_name)

    return fig, ax


def make_popcon_plot(dataset: xr.Dataset, title: str, plot_params: dict, points_params: dict, ax: Axes):
    """Make a plot."""
    from cfspopcon import __version__

    fig = ax.figure
    transform_func = build_transform_function_from_dict(dataset, plot_params)

    coords = plot_params["coords"] if "coords" in plot_params else plot_params["new_coords"]
    legend_elements = dict()

    if "fill" in plot_params:
        # Make a filled plot (max 1 variable)
        subplot_params = plot_params["fill"]
        field = dataset[subplot_params["variable"]]
        units = subplot_params.get("units", field.pint.units)
        field = field.pint.to(units)

        mask = build_mask_from_dict(dataset, plot_params=subplot_params)
        transformed_field = transform_func(field.where(mask))

        im = transformed_field.plot(ax=ax, add_colorbar=False)
        cbar = fig.colorbar(im, ax=ax)

        ax.format_coord = CoordinateFormatter(transformed_field)

        cbar.ax.set_ylabel(
            f"{subplot_params.get('cbar_label', subplot_params['variable'])} {units_to_string(field.pint.units)}",
            rotation=270,
            labelpad=subplot_params.get("labelpad", 15.0),
        )

    if "contour" in plot_params:
        # Overlay contour plots

        for variable, subplot_params in plot_params["contour"].items():
            field = dataset[variable]
            units = subplot_params.get("units", field.pint.units)
            field = field.pint.to(units)

            mask = build_mask_from_dict(dataset, plot_params=subplot_params)
            transformed_field = transform_func(field.where(mask))

            contour_set = transformed_field.plot.contour(
                ax=ax, levels=subplot_params["levels"], colors=[subplot_params["color"]], linestyles=[subplot_params.get("line", "solid")]
            )
            legend_entry = label_contour(
                ax, contour_set, fontsize=subplot_params.get("fontsize", 10.0), format_spec=subplot_params.get("format", "")
            )

            legend_elements[subplot_params["label"]] = legend_entry

    for key, point_params in points_params.items():
        if key not in plot_params.get("points", dict()):
            continue
        point_style = plot_params["points"][key]
        label = point_style.get("label", key)

        point = find_values_at_nearest_point(dataset, point_params)

        plotting_coords = []
        for coord in [coords["x"], coords["y"]]:
            dimension_name = coord["dimension"]

            if dimension_name not in point.coords and f"dim_{dimension_name}" in point.coords:
                dimension_name = f"dim_{dimension_name}"

            requested_units = coord.get("units", "")
            if hasattr(point[dimension_name].pint, "units") and point[dimension_name].pint.units is not None:
                plotting_coords.append(magnitude_in_units(point[dimension_name], requested_units))
            else:
                default_units = Quantity(1.0, default_unit(dimension_name.lstrip("dim_")))
                plotting_coords.append(magnitude_in_units(point[dimension_name] * default_units, requested_units))

        legend_elements[label] = ax.scatter(
            *plotting_coords,
            s=point_style.get("size", None),
            c=point_style.get("color", None),
            marker=point_style.get("marker", None),
        )

    ax.set_title(f"{title} [{__version__}]")
    ax.set_xlabel(coords["x"]["label"])
    ax.set_ylabel(coords["y"]["label"])
    ax.legend(legend_elements.values(), legend_elements.keys(), loc=plot_params.get("legend_loc", "best"))
    plt.tight_layout()

    return fig, ax


def units_to_string(units: Unit) -> str:
    """Given a pint Unit, return a string to represent the unit."""
    dummy_var = Quantity(1.0, units)
    if dummy_var.check("") and np.isclose(dimensionless_magnitude(dummy_var), 1.0):
        return ""
    else:
        return f"[{units:~P}]"


def label_contour(ax: plt.Axes, contour_set: matplotlib.contour.QuadContourSet, format_spec: str = "1.1f", fontsize: float = 10.0):
    """Add in-line labels to contours.

    Returns the first label element, which can be used to construct a legend.
    Works best with contour sets with only one color.

    Inputs:
        ax: the matplotlib axis of the contour plot
        contour_set: the return argument of plt.contour(...)
        format_spec: a format specification for the string formatting
        fontsize: the font size of the label
    """

    def fmt(x):
        return f"{x:{format_spec}}"

    ax.clabel(contour_set, contour_set.levels, inline=True, fmt=fmt, fontsize=fontsize)

    return contour_set.legend_elements()[0][0]
