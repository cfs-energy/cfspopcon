import contextlib
from typing import Any

import pytest
import xarray as xr

from cfspopcon.algorithm_class import Algorithm, CompositeAlgorithm
from cfspopcon.unit_handling import ureg


@pytest.fixture(scope="session")
def BIRDS():
    return [
        "ducks",
        "chooks",
        "all_birds",
    ]


@pytest.fixture(scope="session")
def how_many_birds(BIRDS):
    def count_birds(
        things_that_quack: int, things_that_cluck: int = 2
    ) -> dict[str, Any]:
        ducks = things_that_quack
        chooks = things_that_cluck
        all_birds = ducks + chooks

        local_vars = locals()
        return {key: local_vars[key] for key in BIRDS}

    return Algorithm(function=count_birds, return_keys=BIRDS, skip_registration=True)


@pytest.fixture(scope="session")
def ANIMALS():
    return [
        "sheep",
        "all_birds",
        "all_animals",
    ]


@pytest.fixture(scope="session")
def how_many_animals(ANIMALS):
    def count_animals(
        things_that_baa: int, all_birds: int, new_chickens_per_count: int = 2
    ) -> dict[str, Any]:
        sheep = things_that_baa

        all_birds = all_birds + new_chickens_per_count
        all_animals = sheep + all_birds

        local_vars = locals()
        return {key: local_vars[key] for key in ANIMALS}

    return Algorithm(
        function=count_animals, return_keys=ANIMALS, skip_registration=True
    )


def test_algorithm_kw_only():
    def test(p1, p2, /, p_or_kw, *, kw):
        return {"p2_2": 10}

    with pytest.raises(
        ValueError,
        match="Algorithm only supports functions with keyword arguments.*?POSITIONAL_ONLY parameter p1",
    ):
        _ = Algorithm(function=test, return_keys=["p2_2"], skip_registration=True)


def test_composite_signature(how_many_birds, how_many_animals):
    composite = how_many_birds + how_many_animals
    assert (
        str(composite.run.__signature__)
        == "(things_that_quack: int, things_that_baa: int, things_that_cluck: int = 2, new_chickens_per_count: int = 2) -> xarray.core.dataset.Dataset"
    )


def test_dummy_algorithm(how_many_birds, BIRDS):
    assert how_many_birds.return_keys == BIRDS
    assert how_many_birds.input_keys == ["things_that_quack", "things_that_cluck"]
    assert how_many_birds.required_input_keys == ["things_that_quack"]
    assert how_many_birds.default_keys == ["things_that_cluck"]
    assert how_many_birds.default_values["things_that_cluck"] == 2

    with contextlib.redirect_stdout(None):
        repr(how_many_birds)

    result = how_many_birds.run(things_that_quack=1)
    assert result["all_birds"] == 3
    assert result["ducks"] == 1
    assert result["chooks"] == 2

    ds = xr.Dataset(dict(things_that_quack=3))
    resulting_ds = how_many_birds.update_dataset(ds)
    assert resulting_ds["all_birds"] == 5
    assert resulting_ds["ducks"] == 3
    assert resulting_ds["chooks"] == 2

    ds = xr.Dataset(dict(things_that_cluck=3))
    with pytest.raises(KeyError):
        resulting_ds = how_many_birds.update_dataset(ds)


def test_dummy_composite_algorithm(how_many_birds, BIRDS, how_many_animals, ANIMALS):
    count_the_farm = how_many_birds + how_many_animals

    assert set(count_the_farm.return_keys) == set(BIRDS).union(set(ANIMALS))
    assert set(count_the_farm.input_keys) == {
        "things_that_quack",
        "things_that_cluck",
        "things_that_baa",
        "new_chickens_per_count",
    }
    assert set(count_the_farm.required_input_keys) == {
        "things_that_quack",
        "things_that_baa",
    }

    with contextlib.redirect_stdout(None):
        repr(count_the_farm)

    with pytest.warns():
        result = count_the_farm.run(
            things_that_quack=1, things_that_baa=4, crocodiles=3
        )

    assert result["all_birds"] == 5  # N.b. this includes the new chickens
    assert result["ducks"] == 1
    assert result["chooks"] == 2
    assert result["sheep"] == 4
    assert result["all_animals"] == 9

    ds = xr.Dataset(dict(things_that_quack=1, things_that_baa=4, crocodiles=3))
    ds = count_the_farm.update_dataset(ds)
    assert ds["all_birds"] == 5  # N.b. this includes the new chickens
    assert ds["ducks"] == 1
    assert ds["chooks"] == 2
    assert ds["sheep"] == 4
    assert ds["all_animals"] == 9

    # Here is a subtlety which is worth considering
    # If we call count_the_farm again, how many birds will we have?
    # At first glance, we could have 7, since we add 2 new chickens
    # each time we call count_the_farm
    # However, we actually still have just 5. Why? When we call
    # update_dataset, we first call how_many_birds, which sets
    # all_birds = ducks + chooks, ignoring the number of chickens
    # that we have. Then, all_birds gets passed to how_many_animals
    # where we add two more chickens.
    #
    # A few conclusions
    # 1. You should be very careful when relying on the internal
    #    state of the dataset. The .run() method is more explicit
    #    and can help to catch some of these tricky cases.
    # 2. The algorithm is doing what we asked of it. To be sure,
    #    see test_repeated_dataset_updates
    # 3. Before writing a big unit test, ask yourself how much you
    #    want to commit to the whole farm thing and not just call
    #    your variables something boring like 'a' or 'my_var'
    # 4. Be careful counting your chickens before they've hatched.
    ds = count_the_farm.update_dataset(ds)
    assert ds["all_birds"] == 5


def test_composite_of_composite(how_many_birds: Algorithm, how_many_animals: Algorithm):
    # add of Algorihtm + Composite should flatten into Composite of all Algorithms
    count_the_farm = how_many_birds + CompositeAlgorithm(
        [how_many_animals, how_many_animals]
    )
    # thus the lenght of algorithms should be 3
    assert len(count_the_farm.algorithms) == 3

    ds = xr.Dataset(dict(things_that_quack=1, things_that_baa=4, crocodiles=3))
    ds = count_the_farm.update_dataset(ds)
    assert (
        ds["all_animals"] == 11
    )  # Each time we count how many animals, we get two new chickens

    # test the flattening of composites in __init__ and __add__
    comp = CompositeAlgorithm([how_many_birds, count_the_farm])
    assert len(comp.algorithms) == 4
    comp2 = comp + how_many_birds
    assert len(comp2.algorithms) == 5
    comp3 = comp + comp2
    assert len(comp3.algorithms) == 9

    with pytest.raises(TypeError, match=".*missing arguments.*"):
        _ = count_the_farm.run()

    input_ds = xr.Dataset(dict(things_that_quack=1, things_that_baa=4))
    with pytest.warns(UserWarning, match="The following variables were overridden.*"):
        assert count_the_farm.validate_inputs(
            input_ds,
            quiet=False,
            raise_error_on_missing_inputs=True,
            warn_for_overridden_variables=True,
        )

    input_ds["crocodiles"] = 10
    with pytest.warns(UserWarning, match="Unused input parameters .crocodiles.."):
        ret = count_the_farm.validate_inputs(
            input_ds,
            quiet=False,
            raise_error_on_missing_inputs=True,
            warn_for_overridden_variables=False,
        )

    assert ret is False

    # missing + unused  now
    input_ds = input_ds.drop_vars("things_that_baa")
    with pytest.raises(
        RuntimeError, match="Missing input parameters.*Also had unused.*"
    ):
        count_the_farm.validate_inputs(
            input_ds,
            quiet=False,
            raise_error_on_missing_inputs=True,
            warn_for_overridden_variables=False,
        )

    # missing only
    input_ds = input_ds.drop_vars("crocodiles")
    with pytest.raises(
        RuntimeError, match="Missing input parameters .things_that_baa.."
    ):
        count_the_farm.validate_inputs(
            input_ds,
            quiet=False,
            raise_error_on_missing_inputs=True,
            warn_for_overridden_variables=False,
        )

    wrong_order_comp = how_many_animals + how_many_birds
    with pytest.raises(
        RuntimeError, match="Algorithms out of order. all_birds needed by Algorithm.*"
    ):
        wrong_order_comp.validate_inputs(
            input_ds,
            quiet=False,
            raise_error_on_missing_inputs=True,
            warn_for_overridden_variables=False,
        )


def test_repeated_dataset_updates(how_many_animals):
    ds = xr.Dataset(dict(all_birds=0, things_that_baa=0, new_chickens_per_count=1))
    ds = how_many_animals.update_dataset(ds)
    assert ds["all_animals"] == 1
    ds["new_chickens_per_count"] = 10
    ds = how_many_animals.update_dataset(ds)
    assert ds["all_animals"] == 11

    composite = how_many_animals + how_many_animals
    ds = xr.Dataset(dict(all_birds=0, things_that_baa=0, new_chickens_per_count=11))
    ds = composite.update_dataset(ds)
    assert ds["all_animals"] == 22


def test_composite_of_a_single_algorithm_fails(how_many_birds):
    with pytest.raises(TypeError):
        CompositeAlgorithm(how_many_birds)


def test_single_function_algorithm():
    def dummy_func(a, b):
        """A very descriptive docstring."""
        c, d = b, a
        return c, d

    alg = Algorithm.from_single_function(
        dummy_func,
        return_keys=["c", "d"],
        name="test_dummy",
        skip_unit_conversion=True,
        skip_registration=True,
    )

    result = alg.run(a=1, b=2)
    assert result["c"] == 2
    assert result["d"] == 1

    def in_and_out(average_electron_density):
        return average_electron_density * 2

    alg = Algorithm.from_single_function(
        in_and_out,
        return_keys=["average_electron_density"],
        name="test_dummy_in_and_out",
        skip_registration=True,
    )
    result = alg.run(average_electron_density=1.2 * ureg.n20)
    assert result["average_electron_density"] == 24.0 * ureg.n19


def test_get_algorithm():
    # Pass in Algorithm Enums
    for key in Algorithm.algorithms():
        alg = Algorithm.get_algorithm(key)
        assert alg._name in [f"run_{key}", key, "<lambda>"]

    # Pass in strings instead
    for key in Algorithm.algorithms():
        alg = Algorithm.get_algorithm(key)
        assert alg._name in [f"run_{key}", key, "<lambda>"]
