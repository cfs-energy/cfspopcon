"""Helper functions for dealing with the arrays of species."""

from typing import Any

import xarray as xr

from ...named_options import (
    AtomicSpecies,
)


def make_impurity_concentration_array(
    species_list: list[str | AtomicSpecies] | str | AtomicSpecies,
    concentrations_list: list[float | xr.DataArray] | float | xr.DataArray,
) -> xr.DataArray:
    """Make an xr.DataArray with impurity species and their corresponding concentrations.

    This array should be used as the `impurity_concentration` variable.
    """
    # Convert DataArrays of species into plain lists. This is useful if you want to store AtomicSpecies objects in a dataset.
    if isinstance(species_list, (xr.DataArray)):
        species_list = species_list.values.tolist()
    # Deal with single-value input (not recommended, but avoids a confusing user error)
    if isinstance(species_list, str | AtomicSpecies):
        species_list = [
            species_list,
        ]
    if isinstance(concentrations_list, float | xr.DataArray):
        concentrations_list = [
            concentrations_list,
        ]

    if not len(species_list) == len(concentrations_list):
        raise ValueError(f"Dimension mismatch. Input was species list [{species_list}], concentrations list [{concentrations_list}]")

    array = xr.DataArray()
    for species, concentration in zip(species_list, concentrations_list, strict=False):
        array = extend_impurity_concentration_array(array, species, concentration)

    return array


def make_impurity_concentration_array_from_kwargs(**kwargs: Any) -> xr.DataArray:
    """Make an xr.DataArray with impurity species and their corresponding concentrations, using the format (species1=concentration1, ...)."""
    return make_impurity_concentration_array(list(kwargs.keys()), list(kwargs.values()))


def extend_impurity_concentration_array(
    array: xr.DataArray, species: str | AtomicSpecies, concentration: float | xr.DataArray, add: bool = False
) -> xr.DataArray:
    """Append a new element to the impurity_concentration array.

    This method automatically handles broadcasting.

    If the array already has an entry for `species`, that entry is overwritten by `concentration`.
    Pass `add=True` to instead add `concentration` to the existing entry, which is what the core
    and edge radiator algorithms do so that an intrinsic concentration of the radiator species
    is not lost. An existing entry which is NaN everywhere (as xarray alignment produces when a
    shorter array is merged into a dataset whose `dim_species` already lists the species) is
    treated as absent, so `add=True` writes `concentration` as-is rather than NaN.

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

    if species in array.dim_species:
        # If the input array already has the species that we are writing, we take that entry out of the array
        # (optionally adding it to the concentration), then concatenate the new entry back in below.
        existing = array.sel(dim_species=species, drop=True)
        # An entry which is NaN everywhere is one that xarray alignment padded in, rather than a concentration:
        # for instance, when a dataset's dim_species coordinate already lists the species from an earlier run,
        # so merging a shorter array into it fills the species with NaN. Such an entry is treated as absent.
        if add and not existing.isnull().all():
            concentration = existing + concentration
        array = array.drop_sel(dim_species=species)

    # Concatenate the concentration array in, along the 'dim_species' dimension. xr.concat broadcasts the
    # entries against each other, so the concentration can have dimensions that the array doesn't (and vice versa).
    array = xr.concat((array, concentration.expand_dims("dim_species").assign_coords(dim_species=[species])), dim="dim_species")
    # We sort by the species atomic number, to ensure consistent ordering of the species.
    return array.sortby("dim_species")
