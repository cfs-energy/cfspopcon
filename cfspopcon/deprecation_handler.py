"""Updates popcon inputs from older versions to be compatible with the latest version."""
import warnings
from pathlib import Path
from typing import Any, Union

from .algorithm_class import Algorithm, CompositeAlgorithm
from .named_options import ProfileForm


def handle_deprecated_arguments(input_parameters: dict[str, Any], algorithm: Union[CompositeAlgorithm, Algorithm], points: dict[str, Any], plots: dict[str, Path]) -> tuple[dict[str, Any], Union[CompositeAlgorithm, Algorithm], dict[str, Any], dict[str, Path]]:
    """Updates popcon inputs from older versions to be compatible with the latest version."""
    # Handles new way of switching between profile forms
    if ("density_profile_form" in input_parameters) or ("temp_profile_form" in input_parameters):
        warnings.warn(message="Providing 'density_profile_form' or 'temp_profile_form' is deprecated. Replace 'calc_peaked_profiles' with an algorithm in plasma_profiles such as 'calc_analytic_profiles'.", category=FutureWarning, stacklevel=1)

        for key in ("density_profile_form", "temp_profile_form"):
            if key not in input_parameters:
                raise ValueError(f"Missing {key} in input parameters.")

        density_profile_type = input_parameters.pop("density_profile_form")
        temp_profile_type = input_parameters.pop("temp_profile_form")

        if not density_profile_type == temp_profile_type:
            raise ValueError("Cannot handle unmatched density profile and temp profile types. Use cfspopcon<8.0.0 for this feature.")

        profile_type = density_profile_type

        if profile_type == ProfileForm.analytic:
            new_alg = Algorithm.get_algorithm("calc_peaking_and_analytic_profiles")
        elif profile_type == ProfileForm.prf:
            new_alg = Algorithm.get_algorithm("calc_peaking_and_prf_profiles")
        else:
            raise NotImplementedError(f"Cannot handle a profile type of {profile_type}")

        updated_alg_list = [
            new_alg if alg._name == "calc_peaked_profiles" else alg
            for alg in algorithm.algorithms
        ]
        algorithm = CompositeAlgorithm(
            algorithms=updated_alg_list,
        )


    return input_parameters, algorithm, points, plots
