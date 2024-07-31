"""Helper functions for dealing with the arrays of species."""

from typing import Any, Union

import xarray as xr

from ...named_options import (
    AtomicSpecies,
)


def make_impurity_concentration_array(
    species_list: Union[list[Union[str, AtomicSpecies]], Union[str, AtomicSpecies]],
    concentrations_list: Union[list[Union[float, xr.DataArray]], Union[float, xr.DataArray]],
) -> xr.DataArray:
    """Make an xr.DataArray with impurity species and their corresponding concentrations.

    This array should be used as the `impurity_concentration` variable.
    """
    # Convert DataArrays of species into plain lists. This is useful if you want to store AtomicSpecies objects in a dataset.
    if isinstance(species_list, (xr.DataArray)):
        species_list = species_list.values.tolist()
    # Deal with single-value input (not recommended, but avoids a confusing user error)
    if isinstance(species_list, (str, AtomicSpecies)):
        species_list = [
            species_list,
        ]
    if isinstance(concentrations_list, (float, xr.DataArray)):
        concentrations_list = [
            concentrations_list,
        ]

    if not len(species_list) == len(concentrations_list):
        raise ValueError(f"Dimension mismatch. Input was species list [{species_list}], concentrations list [{concentrations_list}]")

    array = xr.DataArray()
    for species, concentration in zip(species_list, concentrations_list):
        array = extend_impurity_concentration_array(array, species, concentration)

    return array


def make_impurity_concentration_array_from_kwargs(**kwargs: Any) -> xr.DataArray:
    """Make an xr.DataArray with impurity species and their corresponding concentrations, using the format (species1=concentration1, ...)."""
    return make_impurity_concentration_array(list(kwargs.keys()), list(kwargs.values()))


def extend_impurity_concentration_array(
    array: xr.DataArray, species: Union[str, AtomicSpecies], concentration: Union[float, xr.DataArray]
) -> xr.DataArray:
    """Append a new element to the impurity_concentration array.

    This method automatically handles broadcasting.

    N.b. You can also 'extend' an empty array, constructed via xr.DataArray()

    When writing this back into a xr.Dataset, make sure you use xr.merge instead of assignment!
    See: https://github.com/cfs-energy/cfspopcon/pull/66#issuecomment-2256667631
    """
    if isinstance(species, xr.DataArray):
        species = species.item()

    if not isinstance(species, AtomicSpecies):
        species = AtomicSpecies[species.capitalize()]

    if not isinstance(concentration, xr.DataArray):
        concentration = xr.DataArray(concentration)

    if array.ndim == 0:
        # If the input array is empty, then just return the concentration instead of the input array
        return concentration.expand_dims("dim_species").assign_coords(dim_species=[species])
    elif species in array.dim_species:
        # If the input array already has the species that we are writing, we need to carefully write these values
        # into the array
        # First, we make sure that the array is of the correct shape to write concentration in
        array = array.broadcast_like(concentration).copy()
        # Then, we overwrite the values for the species that we are writing, using .loc instead of .sel since
        # we can't assign values with .sel
        array.loc[dict(dim_species=species)] = concentration
        # Finally, we sort by the species atomic number, to ensure consistent ordering of the species.
        return array.sortby("dim_species")
    else:
        # If the input array doesn't have the species that we're writing, we can simply concatenate the
        # concentration array in, along the 'dim_species' dimension.
        array = xr.concat((array, concentration.expand_dims("dim_species").assign_coords(dim_species=[species])), dim="dim_species")
        # We again sort by the species atomic number, to ensure consistent ordering of the species.
        return array.sortby("dim_species")
