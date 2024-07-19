import numpy as np
import pytest

from cfspopcon.formulas.read_atomic_data import AtomicData


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
    species = "helium"
    species = atomic_data.key_to_enum(species)
    kind = AtomicData.NoncoronalLz
    dataset = atomic_data[species]

    Ne = np.logspace(np.log10(dataset.dim_electron_density.min().item() * 1.1), np.log10(dataset.dim_electron_density.max().item() * 0.9))
    Te = np.logspace(np.log10(dataset.dim_electron_temp.min().item() * 1.1), np.log10(dataset.dim_electron_temp.max().item() * 0.9))

    atomic_data.eval_interpolator(electron_density=Ne, electron_temp=Te, kind=kind, species=species, allow_extrapolation=True, ne_tau=1e19)
    atomic_data.eval_interpolator(electron_density=Ne, electron_temp=Te, kind=kind, species=species, allow_extrapolation=False, ne_tau=1e19)

    with pytest.raises(ValueError):
        atomic_data.eval_interpolator(electron_density=Ne, electron_temp=Te, kind=kind, species=species, ne_tau=1e19, grid=False)

    atomic_data.eval_interpolator(
        electron_density=Ne, electron_temp=Te, kind=kind, species=species, ne_tau=1e19, grid=False, coords=dict(rho=np.linspace(0, 1))
    )
