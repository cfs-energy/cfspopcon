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


def test_malformed_units_are_reported_as_unrecognized():
    """Every malformed spelling lands in the same ValueError, whatever pint raises internally."""
    for bad in ["m**", "1000 * liter", "m + s", 5]:
        with pytest.raises(ValueError, match="The following units are not recognized"):
            check_units_are_valid({"bad_variable": bad})


def test_a_variables_file_with_missing_entries_is_refused(tmp_path):
    """An entry without default_units is refused by name; a non-mapping file is refused; an empty file adds nothing."""
    missing = tmp_path / "missing.yaml"
    missing.write_text("some_variable:\n  description:\n  - No units here.\n")
    with pytest.raises(ValueError, match="some_variable"):
        read_default_units_from_file(missing)

    not_a_mapping = tmp_path / "list.yaml"
    not_a_mapping.write_text("- a\n- b\n")
    with pytest.raises(ValueError, match="mapping"):
        read_default_units_from_file(not_a_mapping)

    empty = tmp_path / "empty.yaml"
    empty.write_text("")
    read_default_units_from_file(empty)
