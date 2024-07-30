"""Constructors and helper functions."""

from typing import Any, Union

import xarray as xr

from .named_options import (
    AtomicSpecies,
    ConfinementPowerScaling,
    LambdaQScaling,
    MomentumLossFunction,
    ProfileForm,
    RadiationMethod,
)


def convert_named_options(key: str, val: Any) -> Any:  # noqa: PLR0911
    """Given a 'key' matching a named_option, return the corresponding Enum value."""
    if key in ["temp_profile_form", "density_profile_form"]:
        return ProfileForm[val]
    elif key == "radiated_power_method":
        return RadiationMethod[val]
    elif key == "impurity":
        return AtomicSpecies[val]
    elif key == "impurities":
        return make_impurities_array(list(val.keys()), list(val.values()))
    elif key in ["core_impurity_species", "edge_impurity_species"]:
        return AtomicSpecies[val]
    elif key == "lambda_q_scaling":
        return LambdaQScaling[val]
    elif key == "SOL_momentum_loss_function":
        return MomentumLossFunction[val]
    elif key == "radiation_method":
        return RadiationMethod[val]
    elif key == "confinement_power_scaling":
        return ConfinementPowerScaling[val]
    else:
        # If the key doesn't match, don't convert the value
        return val


def make_impurities_array(
    species_list: Union[list[Union[str, AtomicSpecies]], Union[str, AtomicSpecies]],
    concentrations_list: Union[list[Union[float, xr.DataArray]], Union[float, xr.DataArray]],
) -> xr.DataArray:
    """Make an xr.DataArray with impurity species and their corresponding concentrations.

    This array should be used as the `impurities` variable.
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
        array = extend_impurities_array(array, species, concentration)

    return array


def make_impurities_array_from_kwargs(**kwargs: Any) -> xr.DataArray:
    """Make an xr.DataArray with impurity species and their corresponding concentrations, using the format (species1=concentration1, ...)."""
    return make_impurities_array(list(kwargs.keys()), list(kwargs.values()))


def extend_impurities_array(
    array: xr.DataArray, species: Union[str, AtomicSpecies], concentration: Union[float, xr.DataArray]
) -> xr.DataArray:
    """Append a new element to the impurities array.

    This method automatically handles broadcasting.

    N.b. You can also 'extend' an empty array, constructed via xr.DataArray()
    """
    if isinstance(species, xr.DataArray):
        species = species.item()

    if not isinstance(species, AtomicSpecies):
        species = AtomicSpecies[species.capitalize()]

    if not isinstance(concentration, xr.DataArray):
        concentration = xr.DataArray(concentration)

    if array.ndim == 0:
        return concentration.expand_dims("dim_species").assign_coords(dim_species=[species])
    elif species in array.dim_species:
        array.loc[dict(dim_species=species)] = concentration
        return array.sortby("dim_species")
    else:
        array = xr.concat((array, concentration.expand_dims("dim_species").assign_coords(dim_species=[species])), dim="dim_species")
        return array.sortby("dim_species")
