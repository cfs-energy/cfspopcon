import pytest
from cfspopcon.unit_handling import ureg, UndefinedUnitError
from cfspopcon.unit_handling.default_units import DefaultUnits


def test_read_default_units():
    """Make sure that the default units can be read without error."""
    DefaultUnits.read_units_from_file()


def test_check_units_are_valid():
    valid_dict = dict(value="metres", value2="kg", value3=ureg.eV, value4=ureg.n19)

    DefaultUnits.check_units_are_valid(valid_dict)

    invalid_dict = dict(value4=ureg.n19, value="ducks", value2="chickens", value3=ureg.eV)

    with pytest.raises(UndefinedUnitError):
        DefaultUnits.check_units_are_valid(invalid_dict)
