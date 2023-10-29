import pytest

import cfspopcon


@pytest.mark.filterwarnings("error")
def test_read_atomic_data():
    cfspopcon.atomic_data.read_atomic_data()


@pytest.mark.filterwarnings("error")
def test_read_atomic_data_from_explicit_directory(module_directory):
    cfspopcon.atomic_data.read_atomic_data(module_directory / "atomic_data")


@pytest.mark.filterwarnings("error")
def test_read_atomic_data_from_missing_directory(module_directory):

    with pytest.raises(FileNotFoundError):
        cfspopcon.atomic_data.read_atomic_data(module_directory / "this_doesnt_exist")
