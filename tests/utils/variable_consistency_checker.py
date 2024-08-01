"""A simple tool to ensure consistency between the algorithms, default units and the physics glossary, with a CLI."""

import re
from importlib.resources import as_file, files
from pathlib import Path
from sys import exit
from typing import Any

import click
import yaml

from cfspopcon import Algorithm
from cfspopcon.unit_handling import Quantity


class VariableConsistencyChecker:
    """A class for comparing the keys in algorithms, default units and the physics glossary."""

    def __init__(self) -> None:
        """Read in the keys from the algorithms, default units and the physics glossary."""
        self.glossary, self.glossary_keys = self.read_physics_glossary()
        self.algorithm_keys = self.read_algorithm_keys()
        self.variables_dict, self.variable_keys = self.read_variables_dict()

    def read_physics_glossary(self) -> tuple[dict[str, list[str]], set[str]]:
        """Read the physics glossary file."""
        with as_file(files("cfspopcon").parents[0] / "docs" / "doc_sources" / "physics_glossary.rst") as filepath:  # type:ignore[attr-defined]
            glossary_text = Path(filepath).read_text().splitlines()

        # Find the length of the header
        header_line = 0
        for i, line in enumerate(glossary_text):
            if ":sorted:" in line:
                header_line = i
                break
        if header_line == 0:
            raise RuntimeError("Header not found")
        self._glossary_header = glossary_text[: header_line + 1]

        # Define patterns for identifying blank lines, keys and descriptions.
        pattern_for_key = re.compile(r"^\s{2}\S+\s*$")
        pattern_for_description = re.compile(r"^\s{4}\S.+")
        pattern_for_blank = re.compile(r"^\s*$")

        # Store the results in a dictionary
        glossary: dict[str, list[str]] = dict()

        # Iterate over the glossary, ignoring the header lines
        keys: list[str] = []
        description: list[str] = []

        def check_and_add_to_glossary(keys: list[str], description: list[str]) -> None:
            if keys or description:
                if len(keys) > 1:
                    raise RuntimeError(f"Multiple keys for entry: ({keys}). This must be fixed before proceeding.")
                if len(keys) == 0:
                    raise RuntimeError(f"No key for description '{description}'.")
                glossary[keys[0]] = description

        for line in glossary_text[header_line + 1 :]:
            line_is_blank = pattern_for_blank.match(line)
            line_is_key = pattern_for_key.match(line)
            line_is_description = pattern_for_description.match(line)

            if not (line_is_key or line_is_description):
                # Blank lines indicate new entry
                assert line_is_blank, f"Line didn't match format for key or description, and wasn't blank. Line was: '{line}'"

                check_and_add_to_glossary(keys, description)

                keys = []
                description = []

            if line_is_key:
                keys.append(line.strip())

            if line_is_description:
                description.append(line.strip())

        check_and_add_to_glossary(keys, description)

        return glossary, set(glossary.keys())

    def read_algorithm_keys(self) -> set[str]:
        """Read the algorithm keys."""
        algorithm_keys: set[str] = set()
        for alg in Algorithm.instances.values():
            algorithm_keys.update(alg.input_keys)
            algorithm_keys.update(alg.return_keys)

        return algorithm_keys

    def read_variables_dict(self) -> tuple[dict[str, dict[str, Any]], set[str]]:
        """Read the variables dictionary file."""
        with as_file(files("cfspopcon").joinpath("variables.yaml")) as filepath:
            with open(filepath) as f:
                variables_dict = yaml.safe_load(f)

        return variables_dict, set(variables_dict.keys())

    def run(self, apply_changes: bool = True) -> None: # noqa: PLR0912, PLR0915
        """Check the files and, if apply_changes = True, modify the files in place."""
        success = True

        unused_variable_keys = self.variable_keys - self.algorithm_keys
        unlisted_args = self.algorithm_keys - self.variable_keys

        extra_glossary_keys = (self.glossary_keys - self.variable_keys) - unlisted_args
        undocumented_args = (self.variable_keys - self.glossary_keys) - unused_variable_keys

        linebreak = "\n"
        if len(unused_variable_keys):
            print(
                f"{linebreak}The following keys in the variables dictionary are not used by any algorithm and will be removed:{linebreak}{linebreak.join(list(unused_variable_keys))}{linebreak}"
            )

        if len(unlisted_args):
            print(
                f"{linebreak}The following Algorithm input/output keys are not defined in the variables dictionary and will be added:{linebreak}{linebreak.join(list(unlisted_args))}{linebreak}"
            )

        if len(extra_glossary_keys):
            print(
                f"{linebreak}The following keys in the glossary are not in the variables dictionary and will be removed:{linebreak}{linebreak.join(list(extra_glossary_keys))}{linebreak}"
            )

        if len(undocumented_args):
            print(
                f"{linebreak}The following keys in the variables dictionary are not defined in the glossary and will be added:{linebreak}{linebreak.join(list(undocumented_args))}{linebreak}"
            )

        if len(unused_variable_keys) or len(unlisted_args) or len(extra_glossary_keys) or len(undocumented_args):
            success = False

        if not apply_changes:
            exit(0) if success else exit(1)

        all_keys = sorted((self.variable_keys - unused_variable_keys) | unlisted_args, key=str.lower)
        algs_using_variable: dict[str, list[str]] = {key: [] for key in all_keys}
        algs_setting_variable: dict[str, list[str]] = {key: [] for key in all_keys}
        for alg in Algorithm.instances.values():
            for key in alg.input_keys:
                algs_using_variable[key].append(alg._name) #type:ignore[arg-type]
            for key in alg.return_keys:
                algs_setting_variable[key].append(alg._name) #type:ignore[arg-type]

        new_variables_dict = dict()
        for key in all_keys:
            used_by = algs_using_variable[key]
            set_by = algs_setting_variable[key]

            if key in self.variables_dict:
                default_units = self.variables_dict[key]["default_units"]
                if default_units is not None:
                    default_units = str(Quantity(1.0, default_units).units)
                description = self.variables_dict[key]["description"]
                if key not in self.glossary:
                    print(f"Adding description for '{key}'.{linebreak}New: '{description}'.{linebreak}")
                elif not (description == self.glossary[key]):
                    print(
                        f"Description changing for '{key}'.{linebreak}From: '{self.glossary[key]}'{linebreak}To: '{description}'.{linebreak}"
                    )
                    success = False
            else:
                default_units = None
                description = ["UNKNOWN. Please add a description and default units for this variable."]

            new_variables_dict[key] = dict(
                default_units=default_units,
                description=description,
                set_by=set_by,
                used_by=used_by,
            )

        with as_file(files("cfspopcon").joinpath("variables.yaml")) as filepath:
            with open(filepath, "w") as f:
                yaml.safe_dump(new_variables_dict, f, sort_keys=False)

        unknown_entries = False
        for entry in new_variables_dict.values():
            for description_line in entry["description"]:
                if "UNKNOWN" in description_line:
                    unknown_entries = True

        if unknown_entries:
            print("Description has 'UNKNOWN' variables.yaml. Skipping glossary update.")
            success = False
        else:
            glossary_text = [
                ".. _physics_glossary:",
                "..",
                "  Automatically generated by VariableConsistencyChecker based on variables.yaml. Do not edit this file by hand!" "",
                "",
                "Physics Glossary",
                "==================",
                "",
                ".. glossary::",
                "  :sorted:",
            ]
            for key in new_variables_dict.keys():
                description = new_variables_dict[key]["description"]
                glossary_text += [""]
                glossary_text += [f"  {key}"]
                for line in description:
                    glossary_text += [f"    {line}"]

                with as_file(files("cfspopcon").parents[0] / "docs" / "doc_sources" / "physics_glossary.rst") as filepath:  # type:ignore[attr-defined]
                    filepath.write_text("\n".join(glossary_text))

        exit(0) if success else exit(1)


@click.command()
@click.option("--run", is_flag=True, help="Modifies the checked files in-place.")
def check_variables_cli(run: bool) -> None:
    """Check whether the Algorithm keys, the default_units file and the physics_glossary file are consistent."""
    variable_consistency_checker = VariableConsistencyChecker()
    variable_consistency_checker.run(apply_changes=run)


if __name__ == "__main__":
    check_variables_cli()
