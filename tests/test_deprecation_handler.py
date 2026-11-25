import pytest

from cfspopcon.algorithm_class import registry
from cfspopcon.deprecation_handler import handle_deprecated_arguments
from cfspopcon.named_options import ProfileForm


def test_deprecated_profile_form_requires_composite_algorithm():
    algorithm = registry["calc_plasma_volume"]
    input_parameters = {"density_profile_form": ProfileForm.analytic, "temp_profile_form": ProfileForm.analytic}

    with pytest.warns(FutureWarning), pytest.raises(TypeError, match="CompositeAlgorithm"):
        handle_deprecated_arguments(input_parameters, algorithm, {}, {})
