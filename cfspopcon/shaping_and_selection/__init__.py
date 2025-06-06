"""Functions to post-process a POPCON analysis, such as finding the coordinates matching a particular condition."""

from .line_selection import (
    find_coords_of_contour,
    interpolate_onto_line,
)
from .point_selection import (
    build_mask_from_dict,
    find_coords_of_maximum,
    find_coords_of_minimum,
    find_coords_of_nearest_point,
)
from .transform_coords import (
    build_transform_function_from_dict,
    interpolate_array_onto_new_coords,
    order_dimensions,
)

__all__ = [
    "build_mask_from_dict",
    "build_transform_function_from_dict",
    "find_coords_of_contour",
    "find_coords_of_maximum",
    "find_coords_of_minimum",
    "find_coords_of_nearest_point",
    "interpolate_array_onto_new_coords",
    "interpolate_onto_line",
    "order_dimensions",
]
