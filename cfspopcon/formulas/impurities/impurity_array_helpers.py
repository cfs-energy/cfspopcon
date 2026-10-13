"""Helper functions for dealing with the arrays of species."""

from typing import Any, Literal

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
    impurity_concentration: xr.DataArray,
    species: str | AtomicSpecies,
    concentration: float | xr.DataArray,
    *,
    existing: Literal["replace", "add"] = "replace",
) -> xr.DataArray:
    """Return a copy of `impurity_concentration` with an entry for `species` added or updated.

    `impurity_concentration` is a multi-species `xr.DataArray` indexed along `dim_species`.
    Because adding a new species changes the size of `dim_species` (and may broadcast across
    additional grid dimensions on `concentration`), xarray cannot mutate `impurity_concentration`
    in place: the existing multi-species `DataArray` must be passed in and the updated
    `DataArray` reassigned from the return value::

        impurity_concentration = extend_impurity_concentration_array(
            impurity_concentration, species, concentration, existing="add"
        )

    If `species` is not yet in `impurity_concentration`, it is appended along `dim_species`
    (passing an empty `xr.DataArray()` creates a single-species `impurity_concentration` array).
    If `species` is already in `impurity_concentration`, `existing` controls what happens:

    - `"replace"` (default): the existing entry for `species` is replaced by `concentration`.
    - `"add"`: `concentration` is added to the existing entry for `species`. An existing entry
      that is NaN everywhere (as produced when xarray aligns a shorter array onto a dataset's
      `dim_species` coordinate) is treated as absent, so `concentration` is written as-is.

    When writing the result back into an `xr.Dataset`, use `xr.merge` instead of direct assignment;
    see https://github.com/cfs-energy/cfspopcon/pull/66#issuecomment-2256667631.
    """
    if existing not in ("replace", "add"):
        raise ValueError(f"Unknown existing={existing!r}; expected 'replace' or 'add'")

    if isinstance(species, xr.DataArray):
        species = species.item()

    if not isinstance(species, AtomicSpecies):
        species = AtomicSpecies[species.capitalize()]

    if not isinstance(concentration, xr.DataArray):
        concentration = xr.DataArray(concentration)

    if impurity_concentration.ndim == 0:
        # If the input array is empty, then just return the concentration instead of the input array
        return concentration.expand_dims("dim_species").assign_coords(dim_species=[species])

    if species in impurity_concentration.dim_species:
        # If the input array already has the species that we are writing, we take that entry out of the array
        # (optionally adding it to the concentration), then concatenate the new entry back in below.
        existing_entry = impurity_concentration.sel(dim_species=species, drop=True)
        # An entry which is NaN everywhere is one that xarray alignment padded in, rather than a concentration:
        # for instance, when a dataset's dim_species coordinate already lists the species from an earlier run,
        # so merging a shorter array into it fills the species with NaN. Such an entry is treated as absent.
        if existing == "add" and not existing_entry.isnull().all():
            concentration = existing_entry + concentration
        impurity_concentration = impurity_concentration.drop_sel(dim_species=species)

    # Concatenate the concentration array in, along the 'dim_species' dimension. xr.concat broadcasts the
    # entries against each other, so the concentration can have dimensions that the array doesn't (and vice versa).
    impurity_concentration = xr.concat(
        (impurity_concentration, concentration.expand_dims("dim_species").assign_coords(dim_species=[species])), dim="dim_species"
    )
    # We sort by the species atomic number, to ensure consistent ordering of the species.
    return impurity_concentration.sortby("dim_species")
