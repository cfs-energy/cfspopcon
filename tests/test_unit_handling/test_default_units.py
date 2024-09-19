import pytest

from cfspopcon.unit_handling import ureg
from cfspopcon.unit_handling.default_units import check_units_are_valid, read_default_units_from_file


def test_read_default_units():
    """Make sure that the default units can be read without error."""
    read_default_units_from_file()


def test_check_units_are_valid():
    valid_dict = dict(value="metres", value2="kg", value3=ureg.eV, value4=ureg.n19)

    check_units_are_valid(valid_dict)

    invalid_dict = dict(value4=ureg.n19, value="ducks", value2="chickens", value3=ureg.eV)

    with pytest.raises(ValueError, match="The following units are not recognized.*"):
        check_units_are_valid(invalid_dict)
