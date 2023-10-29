"""Functions to reshape xarrays."""
from collections.abc import Sequence
from typing import Callable, Optional, Union

import numpy as np
import xarray as xr
from scipy.interpolate import griddata  # type:ignore[import]

from cfspopcon.unit_handling import Unit, magnitude


def order_dimensions(
    array: xr.DataArray,
    dims: Sequence[str],
    units: Optional[dict[str, Unit]] = None,
    template: Optional[Union[xr.DataArray, xr.Dataset]] = None,
    order_for_plotting: bool = True,
) -> xr.DataArray:
    """Reorder the dimensions of the array, broadcasting against `template` if necessary.

    This is particularly useful for plotting.
    """
    if template is None:
        template = xr.zeros_like(array)

    processed_dims = []
    for dim in dims:
        # We sometimes add a "dim_" in front of coord to differentiate between
        # variables and coordinates (in particular, coordinates can't have units)
        if dim not in array.dims and f"dim_{dim}" in array.dims:
            dim = f"dim_{dim}"  # noqa: PLW2901

        if dim in array.dims:
            processed_dims.append(dim)
        else:
            # If we've asked for a dimension, but didn't find it in array.dims,
            # look in the template.
            if dim not in template.dims and f"dim_{dim}" in template.dims:
                dim = f"dim_{dim}"  # noqa: PLW2901
            if dim in template.dims:
                # If we find the dimension in the template, broadcast the array
                # to have the correct dimensions.
                array = array.broadcast_like(template[dim])
                processed_dims.append(dim)
            else:
                raise ValueError(f"Array does not have dimension {dim.lstrip('dim_')}")

    if units is not None:
        for dim, processed_dim in zip(dims, processed_dims):
            array[processed_dim] = magnitude(template[dim.lstrip("dim_")].pint.to(units[dim]))

    if order_for_plotting:
        processed_dims = processed_dims[::-1]

    return array.transpose(*processed_dims)


def interpolate_array_onto_new_coords(
    array: xr.DataArray,
    new_coords: dict[str, xr.DataArray],
    resolution: Optional[dict[str, int]] = None,
    coord_min: Optional[dict[str, int]] = None,
    coord_max: Optional[dict[str, int]] = None,
    default_resolution: int = 50,
    max_distance: float = 2.0,
    griddata_method: str = "linear",
) -> xr.DataArray:
    """Take an xarray of values and map it to a grid of new coordinates.

    The input array and the new_coords must share the same coordinates (or be
    able to be broadcast to the same coordinates).

    The new mesh will usually not overlap perfectly with the original mesh, and
    may include regions not included in the original mesh. These regions are set
    to NaN (based on the distance between a sample point and a new-grid point).

    This method works for arbitrary number of new_coords, but works best and makes
    most sense for 2D (two new_coords).
    """
    coords, coord_spacing = dict(), dict()
    for key, coord_array in new_coords.items():
        coords[key], coord_spacing[key] = np.linspace(
            start=coord_min.get(key, magnitude(coord_array.min())) if coord_min else magnitude(coord_array.min()),
            stop=coord_max.get(key, magnitude(coord_array.max())) if coord_max else magnitude(coord_array.max()),
            num=resolution.get(key, default_resolution) if resolution else default_resolution,
            retstep=True,
        )

    mesh_coords = tuple(np.meshgrid(*coords.values(), indexing="ij"))

    broadcast_arrays = xr.broadcast(*(*list(new_coords.values()), array))
    sample_points = tuple([np.ravel(magnitude(broadcast_array)) for broadcast_array in broadcast_arrays[:-1]])
    array = broadcast_arrays[-1]

    interpolated_array = xr.DataArray(
        griddata(
            sample_points,
            np.ravel(magnitude(array)),
            mesh_coords,
            method=griddata_method,
            rescale=True,
        ),
        coords=coords,
        attrs=array.attrs,
    ).pint.quantify(array.pint.units)

    # Calculate the out-of-bounds mask based on distance to nearest sample point
    mesh_shape = [coord.size for coord in coords.values()]
    distance_to_nearest = np.zeros(mesh_shape, dtype=float)
    stacked_samples = np.vstack(sample_points)
    spacing = np.expand_dims(list(coord_spacing.values()), axis=-1)

    for index in np.ndindex(tuple(mesh_shape)):
        # Iterate over each grid point
        mesh_point = [mesh_coords[dimension][index] for dimension in range(len(index))]
        broadcast_mesh_point = np.broadcast_to(np.expand_dims(mesh_point, axis=-1), stacked_samples.shape)

        distance_to_points = ((broadcast_mesh_point - stacked_samples) / spacing) ** 2
        distance_to_nearest[index] = np.min(np.sum(distance_to_points, axis=0))

    clipped_array = interpolated_array.where(distance_to_nearest < max_distance).clip(min=array.min(), max=array.max())

    return clipped_array  # type:ignore[no-any-return]


def build_transform_function_from_dict(dataset: xr.Dataset, plot_params: dict) -> Callable[[xr.DataArray], xr.DataArray]:
    """Build a function which can be called on a field to return an array with transformed coordinates.

    The simplest function is a transpose function, which makes sure that the dimensions are correctly ordered
    for plotting.

    We can also build a more complicated function which transforms the field in terms of other computed fields,
    remapping the field onto new axes.
    """
    if "coords" in plot_params and "new_coords" in plot_params:
        raise ValueError("Can only pass one of 'coords' or 'new_coords'.")

    if "coords" in plot_params:
        xdim = plot_params["coords"]["x"]["dimension"]
        ydim = plot_params["coords"]["y"]["dimension"]
        units = {
            xdim: plot_params["coords"]["x"].get("units"),
            ydim: plot_params["coords"]["y"].get("units"),
        }

        return lambda array: order_dimensions(array, dims=(xdim, ydim), units=units, template=dataset, order_for_plotting=True)

    elif "new_coords" in plot_params:
        new_coords, new_coord_min, new_coord_max, new_coord_res = {}, {}, {}, {}

        for coord in ["y", "x"]:
            new_coord = plot_params["new_coords"][coord]
            new_key = new_coord["dimension"]
            field = dataset[new_key]
            new_coords[new_key] = field.pint.to(new_coord.get("units", field.pint.units))
            if "min" in new_coord:
                new_coord_min[new_key] = new_coord["min"]
            if "max" in new_coord:
                new_coord_max[new_key] = new_coord["max"]
            if "resolution" in new_coord:
                new_coord_res[new_key] = new_coord["resolution"]

        return lambda array: interpolate_array_onto_new_coords(
            array=array,
            new_coords=new_coords,
            resolution=new_coord_res,
            coord_min=new_coord_min,
            coord_max=new_coord_max,
            max_distance=plot_params["new_coords"].get("max_distance", 5.0),
        )

    else:
        raise NotImplementedError("Must provide either 'coords' or 'new_coords' for the transform function.")
