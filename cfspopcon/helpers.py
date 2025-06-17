"""Constructors and helper functions."""

from typing import Any

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
    from .formulas.impurities.impurity_array_helpers import make_impurity_concentration_array

    if key in ["temp_profile_form", "density_profile_form"]:
        return ProfileForm[val]
    elif key == "radiated_power_method":
        return RadiationMethod[val]
    elif key in ["impurity_concentration", "intrinsic_impurity_concentration"]:
        return make_impurity_concentration_array(list(val.keys()), list(val.values()))
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


def get_item(value: Any) -> Any:
    """Check if an object is an xr.DataArray, and if so, return the ".item()" element."""
    if isinstance(value, xr.DataArray):
        return value.item()
    else:
        return value


def get_values(array: Any) -> Any:
    """Check if an object is an xr.DataArray, and if so, return the ".values" element."""
    if isinstance(array, xr.DataArray):
        return array.values
    else:
        return array
