"""How the core and edge radiator algorithms combine with an existing concentration of the radiator species.

Regression tests for https://github.com/cfs-energy/cfspopcon/issues/166.
"""

import copy

import numpy as np
import pytest
import xarray as xr

import cfspopcon
from cfspopcon.deprecation_handler import handle_deprecated_arguments
from cfspopcon.formulas.atomic_data import AtomicData
from cfspopcon.formulas.impurities.impurity_array_helpers import make_impurity_concentration_array_from_kwargs
from cfspopcon.formulas.radiated_power.impurity_radiated_power import calc_impurity_radiated_power
from cfspopcon.input_file_handling import read_case
from cfspopcon.named_options import AtomicSpecies, RadiationMethod
from cfspopcon.unit_handling import magnitude, magnitude_in_units, ureg

INTRINSIC_ARGON = 1e-3
INTRINSIC_TUNGSTEN = 1e-5


@pytest.fixture(scope="module")
def atomic_data():
    return AtomicData()


@pytest.fixture()
def without_argon():
    return make_impurity_concentration_array_from_kwargs(tungsten=INTRINSIC_TUNGSTEN)


@pytest.fixture()
def with_argon():
    return make_impurity_concentration_array_from_kwargs(tungsten=INTRINSIC_TUNGSTEN, argon=INTRINSIC_ARGON)


def _argon(impurity_concentration):
    return magnitude(impurity_concentration.sel(dim_species=AtomicSpecies.Argon))


def _tungsten(impurity_concentration):
    return magnitude(impurity_concentration.sel(dim_species=AtomicSpecies.Tungsten))


@pytest.mark.parametrize("P_seed", [0.0, 1.0], ids=["no seeding", "seeding"])
def test_core_seeding_adds_to_intrinsic_concentration(atomic_data, without_argon, with_argon, P_seed):
    rho = cfspopcon.registry["define_radial_grid"].run()["rho"]
    profiles = dict(
        rho=rho,
        electron_density_profile=1e20 * ureg.m**-3 * (1 - 0.5 * rho**2),
        electron_temp_profile=5.0 * ureg.keV * (1 - 0.8 * rho**2),
        plasma_volume=10.0 * ureg.m**3,
        radiated_power_method=RadiationMethod.Radas,
    )
    seed = cfspopcon.registry["calc_core_seeded_impurity_concentration"]
    kwargs = dict(
        **profiles,
        P_radiation_from_core_seeded_impurity=P_seed * ureg.MW,
        radiated_power_scalar=1.0,
        core_impurity_species=AtomicSpecies.Argon,
        atomic_data=atomic_data,
    )

    without = seed.run(impurity_concentration=without_argon, **kwargs)
    with_intrinsic = seed.run(impurity_concentration=with_argon, **kwargs)
    seeded_only = without["impurity_concentration"]
    seeded_on_top_of_intrinsic = with_intrinsic["impurity_concentration"]

    if P_seed == 0.0:
        assert _argon(seeded_only) == 0.0
    else:
        assert _argon(seeded_only) > 0.0
    assert np.isclose(_argon(seeded_on_top_of_intrinsic), INTRINSIC_ARGON + _argon(seeded_only))
    assert np.isclose(_tungsten(seeded_on_top_of_intrinsic), INTRINSIC_TUNGSTEN)
    assert len(seeded_on_top_of_intrinsic.dim_species) == 2

    # The seeded amount is reported separately, and does not depend on what was already in the array
    core_seeded = magnitude(with_intrinsic["core_seeded_impurity_concentration"])
    assert np.isclose(core_seeded, magnitude(without["core_seeded_impurity_concentration"]))
    assert np.isclose(core_seeded, _argon(seeded_on_top_of_intrinsic) - INTRINSIC_ARGON)
    if P_seed == 0.0:
        assert core_seeded == 0.0

    # The impurity content agrees with the radiated power: what the seeded array radiates is what the
    # intrinsic array radiates, plus the requested seeded power.
    def P_rad(impurity_concentration):
        P = calc_impurity_radiated_power(**profiles, impurity_concentration=impurity_concentration, atomic_data=atomic_data)
        return magnitude_in_units(P.sum(dim="dim_species"), ureg.MW)

    assert np.isclose(P_rad(seeded_on_top_of_intrinsic), P_rad(with_argon) + P_seed)


def test_edge_radiator_sets_total_concentration(atomic_data, without_argon, with_argon):
    kwargs = dict(
        edge_impurity_species=AtomicSpecies.Argon,
        q_parallel=1.0 * ureg.GW / ureg.m**2,
        SOL_power_loss_fraction=0.8,
        target_electron_temp=25.0 * ureg.eV,
        separatrix_electron_temp=100.0 * ureg.eV,
        separatrix_electron_density=1e20 * ureg.m**-3,
        kappa_e0=2600.0 * ureg.W / (ureg.eV**3.5 * ureg.m),
        lengyel_overestimation_factor=4.3,
        edge_impurity_enrichment=5.0,
        atomic_data=atomic_data,
        reference_ne_tau=0.5e17 * ureg.m**-3 * ureg.s,
    )
    seed = cfspopcon.registry["calc_edge_impurity_concentration"]

    without = seed.run(impurity_concentration=without_argon, **kwargs)
    with_intrinsic = seed.run(impurity_concentration=with_argon, **kwargs)

    edge_in_core = magnitude(without["edge_impurity_concentration_in_core"])
    assert edge_in_core > INTRINSIC_ARGON
    assert np.isclose(magnitude(with_intrinsic["edge_impurity_concentration_in_core"]), edge_in_core)
    # The Lengyel concentration is the total for that species, so the array entry is the same with or without
    # an intrinsic concentration, and agrees with edge_impurity_concentration_in_core.
    assert np.isclose(_argon(without["impurity_concentration"]), edge_in_core)
    assert np.isclose(_argon(with_intrinsic["impurity_concentration"]), edge_in_core)
    assert np.isclose(_tungsten(with_intrinsic["impurity_concentration"]), INTRINSIC_TUNGSTEN)


@pytest.mark.filterwarnings("ignore:Not all input parameters were used")
def test_core_radiator_in_intrinsic_impurities_is_idempotent(cases_directory):
    """Re-running the SPARC case on its own output does not add the seeded concentration a second time."""
    input_parameters, algorithm, _, _ = read_case(cases_directory / "SPARC_PRD" / "input.yaml")
    input_parameters, algorithm, _, _ = handle_deprecated_arguments(input_parameters, algorithm, {}, {})
    input_parameters["average_electron_density"] = input_parameters["average_electron_density"][::10]
    input_parameters["average_electron_temp"] = input_parameters["average_electron_temp"][::10]
    input_parameters["minimum_core_radiated_fraction"] = 0.5
    intrinsic_xenon = 1e-5
    input_parameters["intrinsic_impurity_concentration"] = xr.concat(
        [
            input_parameters["intrinsic_impurity_concentration"],
            make_impurity_concentration_array_from_kwargs(xenon=intrinsic_xenon),
        ],
        dim="dim_species",
    ).sortby("dim_species")

    first_run = algorithm.update_dataset(xr.Dataset(input_parameters))
    second_run = algorithm.update_dataset(copy.deepcopy(first_run))

    xenon = first_run["impurity_concentration"].sel(dim_species=AtomicSpecies.Xenon)
    assert (magnitude(xenon) > intrinsic_xenon).any()
    assert (magnitude(xenon) >= intrinsic_xenon).all()
    # The array entry is the intrinsic concentration plus the reported seeded amount
    core_seeded = first_run["core_seeded_impurity_concentration"]
    np.testing.assert_allclose(magnitude(xenon).transpose(*core_seeded.dims), magnitude(core_seeded) + intrinsic_xenon)

    for variable in ["impurity_concentration", "core_seeded_impurity_concentration"]:
        xr.testing.assert_allclose(second_run[variable].transpose(*first_run[variable].dims), first_run[variable])
