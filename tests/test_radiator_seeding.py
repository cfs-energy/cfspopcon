"""The core and edge radiator algorithms add to, rather than replace, an existing concentration of the radiator species.

Regression tests for https://github.com/cfs-energy/cfspopcon/issues/166.
"""

import numpy as np
import pytest

import cfspopcon
from cfspopcon.formulas.atomic_data import AtomicData
from cfspopcon.formulas.impurities.impurity_array_helpers import make_impurity_concentration_array_from_kwargs
from cfspopcon.named_options import AtomicSpecies, RadiationMethod
from cfspopcon.unit_handling import magnitude, ureg

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
    kwargs = dict(
        P_radiation_from_core_seeded_impurity=P_seed * ureg.MW,
        rho=rho,
        electron_density_profile=1e20 * ureg.m**-3 * (1 - 0.5 * rho**2),
        electron_temp_profile=5.0 * ureg.keV * (1 - 0.8 * rho**2),
        plasma_volume=10.0 * ureg.m**3,
        radiated_power_method=RadiationMethod.Radas,
        radiated_power_scalar=1.0,
        core_impurity_species=AtomicSpecies.Argon,
        atomic_data=atomic_data,
    )
    seed = cfspopcon.registry["calc_core_seeded_impurity_concentration"]

    seeded_only = seed.run(impurity_concentration=without_argon, **kwargs)["impurity_concentration"]
    seeded_on_top_of_intrinsic = seed.run(impurity_concentration=with_argon, **kwargs)["impurity_concentration"]

    if P_seed == 0.0:
        assert _argon(seeded_only) == 0.0
    else:
        assert _argon(seeded_only) > 0.0
    assert np.isclose(_argon(seeded_on_top_of_intrinsic), INTRINSIC_ARGON + _argon(seeded_only))
    assert np.isclose(_tungsten(seeded_on_top_of_intrinsic), INTRINSIC_TUNGSTEN)
    assert len(seeded_on_top_of_intrinsic.dim_species) == 2


def test_edge_seeding_adds_to_intrinsic_concentration(atomic_data, without_argon, with_argon):
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

    seeded_only = seed.run(impurity_concentration=without_argon, **kwargs)
    seeded_on_top_of_intrinsic = seed.run(impurity_concentration=with_argon, **kwargs)

    edge_in_core = magnitude(seeded_only["edge_impurity_concentration_in_core"])
    assert edge_in_core > 0.0
    assert np.isclose(_argon(seeded_only["impurity_concentration"]), edge_in_core)
    # The edge radiator's in-core concentration is unaffected by what is already in the array...
    assert np.isclose(magnitude(seeded_on_top_of_intrinsic["edge_impurity_concentration_in_core"]), edge_in_core)
    # ...and is added to the intrinsic concentration rather than replacing it.
    assert np.isclose(_argon(seeded_on_top_of_intrinsic["impurity_concentration"]), INTRINSIC_ARGON + edge_in_core)
    assert np.isclose(_tungsten(seeded_on_top_of_intrinsic["impurity_concentration"]), INTRINSIC_TUNGSTEN)
