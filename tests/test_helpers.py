import pytest
import xarray as xr
import numpy as np

from cfspopcon import named_options
from cfspopcon.helpers import (
    convert_named_options,
)
from cfspopcon.named_options import AtomicSpecies
from cfspopcon.formulas.impurities.impurity_array_helpers import (
    extend_impurity_concentration_array,
    make_impurity_concentration_array,
    make_impurity_concentration_array_from_kwargs,
)


def test_convert_named_options():
    for val, key in (
        (named_options.ProfileForm.analytic, "density_profile_form"),
        (named_options.RadiationMethod.Radas, "radiated_power_method"),
        (named_options.AtomicSpecies.Neon, "edge_impurity_species"),
        (named_options.AtomicSpecies.Xenon, "core_impurity_species"),
        (named_options.LambdaQScaling.EichRegression15, "lambda_q_scaling"),
        (named_options.MomentumLossFunction.KotovReiter, "SOL_momentum_loss_function"),
    ):
        assert convert_named_options(key=key, val=val.name) == val

    assert convert_named_options(key="ducks", val=23.0) == 23.0

    da = convert_named_options(key="impurity_concentration", val=dict(tungsten=1e-5, helium=1e-2))

    assert da.sel(dim_species=named_options.AtomicSpecies.Tungsten) == 1e-5
    assert da.sel(dim_species=named_options.AtomicSpecies.Helium) == 1e-2


def test_impurity_array_helpers():
    array = xr.DataArray([[1, 2], [3, 4]], coords=dict(a=[1, 2], b=[3, 5]))

    make_impurity_concentration_array(xr.DataArray("tungsten"), array)

    from_lists = make_impurity_concentration_array([named_options.AtomicSpecies.Tungsten, "Xenon"], [array, 2 * array])
    from_kwargs = make_impurity_concentration_array_from_kwargs(tungsten=array, xenon=2 * array)

    assert from_lists.equals(from_kwargs)

    from_extension = make_impurity_concentration_array(["tungsten"], [array])
    from_extension = extend_impurity_concentration_array(from_extension, "xenon", 2 * array)

    assert from_extension.equals(from_kwargs)

    with pytest.raises(ValueError):
        from_lists = make_impurity_concentration_array("Xenon", [array, 2 * array, 3 * array])
    with pytest.raises(ValueError):
        from_lists = make_impurity_concentration_array(["Xenon", "tungsten"], [array])

    array2 = make_impurity_concentration_array_from_kwargs(
        helium=xr.DataArray([0.1, 0.2], dims=("a")),
        tungsten=xr.DataArray([0.3, 0.4], dims=("b")),
    )

    ds = xr.Dataset(data_vars=dict(array1=array, array2=array2))

    ds["array2"] = extend_impurity_concentration_array(ds["array2"], "nitrogen", 1e-3)

    assert np.isclose(ds["array2"].sel(dim_species=AtomicSpecies.Helium).isel(a=0, b=1).item(), 0.1)
    assert np.isclose(ds["array2"].sel(dim_species=AtomicSpecies.Tungsten).isel(a=1, b=1).item(), 0.4)
