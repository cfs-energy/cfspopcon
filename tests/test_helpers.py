import pytest
import xarray as xr

from cfspopcon import named_options
from cfspopcon.helpers import (
    convert_named_options,
    extend_impurities_array,
    make_impurities_array,
    make_impurities_array_from_kwargs,
)


def test_convert_named_options():
    for val, key in (
        (named_options.Algorithms.predictive_popcon, "algorithms"),
        (named_options.ConfinementScaling.ITER98y2, "energy_confinement_scaling"),
        (named_options.ProfileForm.analytic, "profile_form"),
        (named_options.RadiationMethod.Radas, "radiated_power_method"),
        (named_options.ReactionType.DT, "fusion_reaction"),
        (named_options.AtomicSpecies.Neon, "impurity"),
        (named_options.AtomicSpecies.Xenon, "core_radiator"),
        (named_options.LambdaQScaling.EichRegression15, "lambda_q_scaling"),
        (named_options.MomentumLossFunction.KotovReiter, "SOL_momentum_loss_function"),
    ):
        assert convert_named_options(key=key, val=val.name) == val

    assert convert_named_options(key="ducks", val=23.0) == 23.0

    da = convert_named_options(key="impurities", val=dict(tungsten=1e-5, helium=1e-2))

    assert da.sel(dim_species=named_options.AtomicSpecies.Tungsten) == 1e-5
    assert da.sel(dim_species=named_options.AtomicSpecies.Helium) == 1e-2


def test_impurity_array_helpers():
    array = xr.DataArray([[1, 2], [3, 4]], coords=dict(a=[1, 2], b=[3, 5]))

    make_impurities_array(xr.DataArray("tungsten"), array)

    from_lists = make_impurities_array([named_options.AtomicSpecies.Tungsten, "Xenon"], [array, 2 * array])
    from_kwargs = make_impurities_array_from_kwargs(tungsten=array, xenon=2 * array)

    assert from_lists.equals(from_kwargs)

    from_extension = make_impurities_array(["tungsten"], [array])
    from_extension = extend_impurities_array(from_extension, "xenon", 2 * array)

    assert from_extension.equals(from_kwargs)

    with pytest.raises(ValueError):
        from_lists = make_impurities_array("Xenon", [array, 2 * array, 3 * array])
    with pytest.raises(ValueError):
        from_lists = make_impurities_array(["Xenon", "tungsten"], [array])
