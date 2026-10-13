import numpy as np
import pytest
import xarray as xr
from utils.variable_consistency_checker import VariableConsistencyChecker

from cfspopcon import named_options
from cfspopcon.formulas.impurities.impurity_array_helpers import (
    extend_impurity_concentration_array,
    make_impurity_concentration_array,
    make_impurity_concentration_array_from_kwargs,
)
from cfspopcon.helpers import (
    convert_named_options,
)
from cfspopcon.named_options import AtomicSpecies
from cfspopcon.unit_handling import magnitude, ureg


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


def test_extend_impurity_concentration_array_existing_species():
    array = make_impurity_concentration_array_from_kwargs(tungsten=1e-5, argon=1e-3)

    # By default an existing entry is overwritten, and the other entries are untouched
    overwritten = extend_impurity_concentration_array(array, "argon", 2e-3)
    assert np.isclose(overwritten.sel(dim_species=AtomicSpecies.Argon).item(), 2e-3)
    assert np.isclose(overwritten.sel(dim_species=AtomicSpecies.Tungsten).item(), 1e-5)
    assert list(overwritten.dim_species.values) == [AtomicSpecies.Argon, AtomicSpecies.Tungsten]

    # existing="replace" is the default, and can be passed explicitly
    assert extend_impurity_concentration_array(array, "argon", 2e-3, existing="replace").equals(overwritten)

    # With existing="add" the new concentration is added to the existing entry
    added = extend_impurity_concentration_array(array, AtomicSpecies.Argon, 2e-3, existing="add")
    assert np.isclose(added.sel(dim_species=AtomicSpecies.Argon).item(), 3e-3)
    assert np.isclose(added.sel(dim_species=AtomicSpecies.Tungsten).item(), 1e-5)

    # Anything else is rejected rather than silently treated as one of the two
    with pytest.raises(ValueError, match="expected 'replace' or 'add'"):
        extend_impurity_concentration_array(array, "argon", 2e-3, existing="overwrite")  # type: ignore[arg-type]

    # existing="add" for a species which isn't in the array behaves like a plain extension
    added_new = extend_impurity_concentration_array(array, "neon", 4e-3, existing="add")
    assert np.isclose(added_new.sel(dim_species=AtomicSpecies.Neon).item(), 4e-3)
    assert len(added_new.dim_species) == 3

    # The only species in the array can be overwritten or added to
    single = make_impurity_concentration_array("argon", 1e-3)
    assert np.isclose(extend_impurity_concentration_array(single, "argon", 2e-3).item(), 2e-3)
    assert np.isclose(extend_impurity_concentration_array(single, "argon", 2e-3, existing="add").item(), 3e-3)

    # Broadcasting against a concentration with extra dimensions, with and without units
    for concentration in (xr.DataArray([1e-3, 2e-3], dims="x"), xr.DataArray([1e-3, 2e-3], dims="x") * ureg.dimensionless):
        broadcast = extend_impurity_concentration_array(array, "argon", concentration, existing="add")
        assert set(broadcast.dims) == {"dim_species", "x"}
        np.testing.assert_allclose(magnitude(broadcast.sel(dim_species=AtomicSpecies.Argon)), [2e-3, 3e-3])
        np.testing.assert_allclose(magnitude(broadcast.sel(dim_species=AtomicSpecies.Tungsten)), [1e-5, 1e-5])


def test_extend_impurity_concentration_array_add_to_nan_entry():
    # Merging into a dataset whose dim_species already lists a species pads that species with NaN. This is what
    # happens to intrinsic_impurity_concentration on a second update_dataset pass, once the radiators have been added.
    padded = make_impurity_concentration_array_from_kwargs(tungsten=1e-5, argon=1e-3).reindex(
        dim_species=[AtomicSpecies.Argon, AtomicSpecies.Xenon, AtomicSpecies.Tungsten]
    )
    assert padded.sel(dim_species=AtomicSpecies.Xenon).isnull().all()

    # An all-NaN entry is treated as absent, so the new concentration is written rather than NaN
    added = extend_impurity_concentration_array(padded, "xenon", xr.DataArray([1e-4, 2e-4], dims="x"), existing="add")
    np.testing.assert_allclose(added.sel(dim_species=AtomicSpecies.Xenon), [1e-4, 2e-4])
    np.testing.assert_allclose(added.sel(dim_species=AtomicSpecies.Argon), [1e-3, 1e-3])

    # A partially-NaN entry is a real concentration which is invalid at some points, and the NaN propagates
    partial = make_impurity_concentration_array_from_kwargs(tungsten=1e-5, argon=xr.DataArray([1e-3, np.nan], dims="x"))
    added = extend_impurity_concentration_array(partial, "argon", 2e-3, existing="add")
    np.testing.assert_allclose(added.sel(dim_species=AtomicSpecies.Argon), [3e-3, np.nan])


def test_variable_consistency_checker_normalize_description():
    assert VariableConsistencyChecker.normalize_description("single line") == ["single line"]
    assert VariableConsistencyChecker.normalize_description("first line\nsecond line") == ["first line", "second line"]
    assert VariableConsistencyChecker.normalize_description(["first line", "second line"]) == ["first line", "second line"]
