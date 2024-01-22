import matplotlib.pyplot as plt
import numpy as np
import pytest

from cfspopcon import plotting
from cfspopcon.unit_handling import ureg


def test_coordinate_formatter(z):
    formatter = plotting.CoordinateFormatter(z)

    x_test = 1.23
    y_test = -3.45
    ret_string = formatter(x_test, y_test)

    x_string, y_string, z_string = ret_string.split(",")

    assert np.isclose(float(x_string.split("=")[1]), x_test)
    assert np.isclose(float(y_string.split("=")[1]), y_test)
    # Nearest-neighbor interpolation is not particularly accurate.
    assert np.isclose(float(z_string.split("=")[1]), x_test + y_test, atol=0.01)


@pytest.mark.filterwarnings("error")
def test_label_contour(z):
    # Make sure that the label contour functionality runs through.
    _, ax = plt.subplots()
    CS = z.plot.contour(ax=ax, colors=["r"])

    contour_labels = dict()
    contour_labels["z"] = plotting.label_contour(ax=ax, contour_set=CS, format_spec="3.2f", fontsize=12)

    ax.legend(contour_labels.values(), contour_labels.keys())

    plt.close()


def test_units_to_string():
    assert plotting.units_to_string(ureg.dimensionless) == ""
    assert plotting.units_to_string(ureg.m) == "[m]"
    assert plotting.units_to_string(ureg.percent) == "[percent]"
