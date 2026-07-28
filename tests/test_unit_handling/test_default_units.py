import pytest

from cfspopcon.unit_handling import ureg
from cfspopcon.unit_handling.default_units import (
    check_units_are_valid,
    default_unit,
    default_units_map,
    extend_default_units_map,
    read_default_units_from_file,
    reset_default_units,
)


@pytest.fixture()
def restore_default_units():
    """Put the default units map back, so a test may load a file of its own into it."""
    saved = default_units_map()
    yield
    reset_default_units()
    extend_default_units_map(saved)


def test_read_default_units():
    """Make sure that the default units can be read without error."""
    read_default_units_from_file()


def test_check_units_are_valid():
    valid_dict = dict(value="metres", value2="kg", value3=ureg.eV, value4=ureg.n19)

    check_units_are_valid(valid_dict)

    invalid_dict = dict(value4=ureg.n19, value="ducks", value2="chickens", value3=ureg.eV)

    with pytest.raises(ValueError, match="The following units are not recognized.*"):
        check_units_are_valid(invalid_dict)


PLUGIN_VARIABLES = """\
plugin_length:
  default_units: meter
  description:
  - A length defined by a plugin
plugin_selector:
  default_units: null
  description:
  - A class-typed switch, deliberately not a unitful quantity
"""


def test_read_default_units_from_a_given_file(tmp_path, restore_default_units):
    """A package may ship its own variables.yaml and load it, rather than hand-writing a units dict."""
    path = tmp_path / "variables.yaml"
    path.write_text(PLUGIN_VARIABLES)

    read_default_units_from_file(path)

    assert default_unit("plugin_length") == "meter"
    # None means "not a unitful quantity", and must survive as None rather than becoming dimensionless.
    assert default_unit("plugin_selector") is None
    # cfspopcon's own entries are untouched by loading another file.
    assert default_unit("major_radius") == "meter"


def test_an_invalid_unit_in_a_given_file_names_the_key(tmp_path, restore_default_units):
    """The file is validated the same way an inline units dict is."""
    path = tmp_path / "variables.yaml"
    path.write_text("plugin_bad:\n  default_units: not_a_unit\n")

    with pytest.raises(ValueError, match="plugin_bad: not_a_unit"):
        read_default_units_from_file(path)


def test_check_units_are_valid_skips_none():
    """None is not validated as a unit, so it cannot be coerced to dimensionless."""
    check_units_are_valid(dict(a_class=None, a_length="meter"))
