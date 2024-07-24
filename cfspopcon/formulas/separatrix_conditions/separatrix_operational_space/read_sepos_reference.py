"""Reads results from the original SepOS paper (:cite:`Eich_2021`) as a reference."""

from importlib.resources import as_file, files

import yaml


def read_AUG_SepOS_reference() -> dict[str, dict[str, list[float]]]:
    """Reads the arrays of reference values from the AUG_SepOS_reference.yml file."""
    with as_file(
        files(
            "cfspopcon.formulas.separatrix_conditions.separatrix_operational_space"
        ).joinpath("AUG_SepOS_reference.yml")
    ) as filepath:
        with open(filepath) as f:
            data = yaml.safe_load(f)

    return data  # type:ignore[no-any-return]
