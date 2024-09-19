from pathlib import Path

import pytest
import yaml

from cfspopcon.algorithm_class import Algorithm, CompositeAlgorithm
from cfspopcon.input_file_handling import read_case


@pytest.fixture
def test_dict():
    return dict(Q=1.0)


def test_blank_dictionary(test_dict):
    read_case(test_dict)


def test_blank_file(test_dict, tmp_path):
    with open(tmp_path / "input.yaml", "w") as file:
        yaml.dump(test_dict, file)

    read_case(tmp_path)


def test_blank_file_with_another_suffix(test_dict, tmp_path):
    with open(tmp_path / "another.filetype", "w") as file:
        yaml.dump(test_dict, file)

    read_case(tmp_path / "another.filetype")


def test_algorithm_read_single_from_input_file():
    test_dict = dict(algorithms=["read_atomic_data"])

    repr_d, algorithm, points, plots = read_case(test_dict)

    assert isinstance(algorithm, Algorithm)


def test_algorithm_read_multiple_from_input_file():
    test_dict = dict(algorithms=["read_atomic_data", "set_up_impurity_concentration_array"])

    repr_d, algorithm, points, plots = read_case(test_dict)

    assert isinstance(algorithm, CompositeAlgorithm)


def test_read_example_input_file():
    example_case = Path(__file__).parents[1] / "example_cases" / "SPARC_PRD" / "input.yaml"

    read_case(example_case)
