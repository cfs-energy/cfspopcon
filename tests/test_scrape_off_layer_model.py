import numpy as np
import pytest

from cfspopcon import formulas
from cfspopcon.named_options import LambdaQScaling
from cfspopcon.unit_handling import ureg

lambda_q_tests = {
    LambdaQScaling.Brunner: 0.4332283874128845 * ureg.mm,
    LambdaQScaling.EichRegression14: 0.20533809707365488 * ureg.mm,
    LambdaQScaling.EichRegression15: 0.34842497310813536 * ureg.mm,
    LambdaQScaling.EichRegression9: 0.5865460692254366 * ureg.mm,
}


@pytest.fixture()
def average_total_pressure():
    return 732028.9793 * ureg.Pa


@pytest.fixture()
def power_crossing_separatrix():
    return 25.57417052 * ureg.MW


@pytest.fixture()
def major_radius():
    return 1.85 * ureg.m


@pytest.fixture()
def B_pol_out_mid():
    return 3.052711915 * ureg.T


@pytest.fixture()
def inverse_aspect_ratio():
    return 0.3081000000


@pytest.fixture()
def magnetic_field_on_axis():
    return 12.20000000 * ureg.T


@pytest.fixture()
def q_star():
    return 3.290275716


@pytest.fixture()
def lambda_q_factor():
    return 1.23


@pytest.mark.parametrize(["scaling", "result"], lambda_q_tests.items(), ids=[key.name for key in lambda_q_tests.keys()])
def test_lambda_q_scalings(
    scaling,
    result,
    average_total_pressure,
    power_crossing_separatrix,
    major_radius,
    B_pol_out_mid,
    inverse_aspect_ratio,
    magnetic_field_on_axis,
    q_star,
    lambda_q_factor,
):
    lambda_q = formulas.scrape_off_layer.calc_lambda_q(
        lambda_q_scaling=scaling,
        average_total_pressure=average_total_pressure,
        power_crossing_separatrix=power_crossing_separatrix,
        major_radius=major_radius,
        B_pol_out_mid=B_pol_out_mid,
        inverse_aspect_ratio=inverse_aspect_ratio,
        magnetic_field_on_axis=magnetic_field_on_axis,
        q_star=q_star,
        lambda_q_factor=lambda_q_factor,
    )

    assert np.isclose(lambda_q, result)


@pytest.mark.parametrize(["scaling", "result"], lambda_q_tests.items(), ids=[key.name for key in lambda_q_tests.keys()])
def test_lambda_q_scalings_with_algorithms(
    scaling,
    result,
    average_total_pressure,
    power_crossing_separatrix,
    major_radius,
    B_pol_out_mid,
    inverse_aspect_ratio,
    magnetic_field_on_axis,
    q_star,
    lambda_q_factor,
):
    if scaling == LambdaQScaling.Brunner:
        lambda_q = formulas.scrape_off_layer.lambda_q.calc_lambda_q_with_brunner(
            average_total_pressure=average_total_pressure, lambda_q_factor=lambda_q_factor
        )
    elif scaling == LambdaQScaling.EichRegression14:
        lambda_q = formulas.scrape_off_layer.lambda_q.calc_lambda_q_with_eich_regression_14(
            B_pol_out_mid=B_pol_out_mid,
            lambda_q_factor=lambda_q_factor,
        )
    elif scaling == LambdaQScaling.EichRegression15:
        lambda_q = formulas.scrape_off_layer.lambda_q.calc_lambda_q_with_eich_regression_15(
            power_crossing_separatrix=power_crossing_separatrix,
            major_radius=major_radius,
            B_pol_out_mid=B_pol_out_mid,
            inverse_aspect_ratio=inverse_aspect_ratio,
            lambda_q_factor=lambda_q_factor,
        )
    elif scaling == LambdaQScaling.EichRegression9:
        lambda_q = formulas.scrape_off_layer.lambda_q.calc_lambda_q_with_eich_regression_9(
            magnetic_field_on_axis=magnetic_field_on_axis,
            q_star=q_star,
            power_crossing_separatrix=power_crossing_separatrix,
            lambda_q_factor=lambda_q_factor,
        )
    else:
        raise NotImplementedError(f"Add the algorithm for {scaling.name}.")

    assert np.isclose(lambda_q, result)
