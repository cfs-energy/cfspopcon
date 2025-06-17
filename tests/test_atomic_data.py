import numpy as np
import xarray as xr
import pytest

from cfspopcon.formulas.atomic_data import AtomicData
from cfspopcon.named_options import AtomicSpecies
from cfspopcon.unit_handling import magnitude_in_units, dimensionless_magnitude, Quantity, ureg


def test_read_atomic_data():
    AtomicData()


def test_read_atomic_data_from_explicit_directory(repository_directory):
    AtomicData(repository_directory / "radas_dir")


def test_read_atomic_data_from_missing_directory(repository_directory):
    with pytest.raises(FileNotFoundError):
        AtomicData(repository_directory / "this_doesnt_exist")


def test_read_atomic_data_with_no_output(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    with pytest.raises(FileNotFoundError):
        AtomicData(d)


@pytest.fixture(scope="session")
def atomic_data():
    return AtomicData()


def test_key_to_enum(atomic_data):
    atomic_data["hydrogen"]


def test_eval_interpolator(atomic_data):
    species = AtomicSpecies.Helium
    ne_tau = 0.5e17

    interpolator = atomic_data.noncoronal_Lz_interpolators[(species, ne_tau)]

    def logspace(min_val, max_val, num=50):
        return np.logspace(np.log10(min_val), np.log10(max_val), num=num)

    n_dens = 47
    n_temp = 32

    electron_density = xr.DataArray(
        (vals := logspace(interpolator.min_density, interpolator.max_density, num=n_dens)) * interpolator.electron_density_units,
        coords=dict(dim_electron_density=vals),
    )

    electron_temp = xr.DataArray(
        (vals := logspace(interpolator.min_temp, interpolator.max_temp, num=n_temp)) * interpolator.electron_temp_units,
        coords=dict(dim_electron_temp=vals),
    )

    grid_eval = interpolator.grid_eval(electron_density, electron_temp)

    assert grid_eval.sizes["dim_electron_density"] == n_dens
    assert grid_eval.sizes["dim_electron_temp"] == n_temp

    t_dens = 39
    t_temp = 12

    scalar_eval = interpolator.eval(electron_density.isel(dim_electron_density=t_dens), electron_temp.isel(dim_electron_temp=t_temp))

    assert np.isclose(
        magnitude_in_units(grid_eval.isel(dim_electron_density=t_dens, dim_electron_temp=t_temp), interpolator.units),
        magnitude_in_units(scalar_eval, interpolator.units),
    )

    unitless_eval = interpolator(
        magnitude_in_units(electron_density.isel(dim_electron_density=t_dens), interpolator.electron_density_units),
        magnitude_in_units(electron_temp.isel(dim_electron_temp=t_temp), interpolator.electron_temp_units),
    )

    assert np.isclose(
        magnitude_in_units(grid_eval.isel(dim_electron_density=t_dens, dim_electron_temp=t_temp), interpolator.units),
        unitless_eval,
    )

    # Check that the interpolator works up to the boundary
    val00 = interpolator(
        magnitude_in_units(electron_density.isel(dim_electron_density=0), interpolator.electron_density_units),
        magnitude_in_units(electron_temp.isel(dim_electron_temp=0), interpolator.electron_temp_units),
    )

    # Check that an AssertionError is raised if extrapolating off the grid with allow_extrap = False (default)
    with pytest.raises(AssertionError):
        interpolator(
            magnitude_in_units(electron_density.isel(dim_electron_density=0) * 0.9, interpolator.electron_density_units),
            magnitude_in_units(electron_temp.isel(dim_electron_temp=0), interpolator.electron_temp_units),
            allow_extrap=False,
        )

    # Check that nearest-neighbour extrapolation is used if extrapolating off the grid with allow_extrap = False (default)
    test00 = interpolator(
        magnitude_in_units(electron_density.isel(dim_electron_density=0) * 0.9, interpolator.electron_density_units),
        magnitude_in_units(electron_temp.isel(dim_electron_temp=0), interpolator.electron_temp_units),
        allow_extrap=True,
    )
    assert np.isclose(val00, test00)

    test_electron_density = electron_density.isel(dim_electron_density=t_dens)
    test_electron_temp = electron_temp.isel(dim_electron_temp=t_temp)

    n_pts = 32
    vector_dim = np.arange(n_pts)
    ones = xr.DataArray(np.ones_like(vector_dim), coords=dict(vector=vector_dim))

    vector_eval = interpolator.vector_eval(test_electron_density * ones, test_electron_temp * ones)

    assert np.allclose(magnitude_in_units(vector_eval, interpolator.units), magnitude_in_units(scalar_eval, interpolator.units))


@pytest.mark.parametrize("species", ["helium", AtomicSpecies.Nitrogen], ids=["str", "AtomicSpecies"])
@pytest.mark.parametrize("ne_tau", [0.5e17, Quantity(0.5, ureg.ms * ureg.n20)], ids=["float", "Quantity"])
def test_get_functions(atomic_data, species, ne_tau):
    atomic_data.get_coronal_Lz_interpolator(species)
    atomic_data.get_coronal_Z_interpolator(species)
    atomic_data.get_noncoronal_Lz_interpolator(species, ne_tau)
    atomic_data.get_noncoronal_Z_interpolator(species, ne_tau)


@pytest.mark.parametrize("species", ["helium", AtomicSpecies.Nitrogen], ids=["str", "AtomicSpecies"])
@pytest.mark.parametrize("ne_tau", [0.75e17, Quantity(0.05, ureg.ms * ureg.n20)], ids=["float", "Quantity"])
def test_get_functions_with_missing_ne_tau(atomic_data, species, ne_tau):
    with pytest.warns(UserWarning):
        atomic_data.get_noncoronal_Lz_interpolator(species, ne_tau)

    with pytest.warns(UserWarning):
        atomic_data.get_noncoronal_Z_interpolator(species, ne_tau)

    with pytest.raises(KeyError):
        atomic_data.get_noncoronal_Lz_interpolator(species, ne_tau, ne_tau_rel_tolerance=1e-2)

    with pytest.raises(KeyError):
        atomic_data.get_noncoronal_Z_interpolator(species, ne_tau, ne_tau_rel_tolerance=1e-2)


def test_get_radas_version(atomic_data):
    atomic_data._check_radas_version(atomic_data.radas_version)

    with pytest.warns(UserWarning):
        atomic_data._check_radas_version("versions won't have spaces in them.")

    assert atomic_data.radas_version == atomic_data["hydrogen"].radas_version == atomic_data["helium"].radas_version
