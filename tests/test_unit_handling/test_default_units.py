import pytest
from cfspopcon.unit_handling import ureg, UndefinedUnitError
from cfspopcon.unit_handling.default_units import read_default_units, check_units_are_valid


def test_read_default_units():
    """Make sure that the default units can be read without error."""
    read_default_units()


def test_check_units_are_valid():
    valid_dict = dict(value="metres", value2="kg", value3=ureg.eV, value4=ureg.n19)

    check_units_are_valid(valid_dict)

    invalid_dict = dict(value4=ureg.n19, value="ducks", value2="chickens", value3=ureg.eV)

    with pytest.raises(UndefinedUnitError):
        check_units_are_valid(invalid_dict)
