"""Test the pint-xarray wraps_ufunc decorator."""

import warnings

import numpy as np
import pytest
import xarray as xr

from cfspopcon.unit_handling import (
    UnitStrippedWarning,
    dimensionless_magnitude,
    ureg,
    wraps_ufunc,
)


def check_equal(a, b, **kwargs):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        ratio = a / b
        ratio[(a == 0) & (b == 0)] = 1.0

    return np.allclose(dimensionless_magnitude(ratio), 1.0, **kwargs)


@pytest.mark.filterwarnings("error")
def test_wraps_simple():
    @wraps_ufunc(return_units=dict(doubled=None), input_units=dict(x=None))
    def simple_function(x):
        return 2 * x

    x_test = xr.DataArray(np.linspace(0.0, 100.0))
    assert check_equal(2.0 * x_test, simple_function(x_test))
    assert isinstance(simple_function(x_test), xr.DataArray)

    x_test_2 = np.linspace(0.0, 100.0)
    assert check_equal(2.0 * x_test_2, simple_function(x_test_2))


@pytest.mark.filterwarnings("error")
def test_wraps_with_too_many_input_units():
    with pytest.raises(ValueError):

        @wraps_ufunc(
            return_units=dict(result=ureg.m), input_units=dict(a=ureg.m, b=ureg.mm)
        )
        def in_and_out(a):
            return a


@pytest.mark.filterwarnings("error")
def test_wraps_with_too_many_output_units():
    @wraps_ufunc(return_units=dict(a=ureg.m, b=ureg.m), input_units=dict(a=ureg.m))
    def in_and_out(a):
        return a

    with pytest.raises(ValueError):
        in_and_out(ureg.Quantity(1.2, ureg.m))


@pytest.mark.filterwarnings("error")
def test_wraps_with_wrong_arguments():
    with pytest.raises(ValueError):

        @wraps_ufunc(return_units=dict(result=ureg.m), input_units=dict(b=ureg.m))
        def in_and_out(a):
            return a


@pytest.mark.filterwarnings("error")
def test_jumbled_inputs():
    @wraps_ufunc(return_units=dict(result=ureg.m), input_units=dict(a=ureg.m, b=ureg.m))
    def add_together(a, b):
        return a + b

    with pytest.raises(RuntimeError):
        add_together(ureg.Quantity(7.0, ureg.mm), a=ureg.Quantity(3.0, ureg.m))


@pytest.mark.filterwarnings("error")
def test_pass_as_kwargs():
    @wraps_ufunc(
        return_units=dict(result=ureg.m),
        input_units=dict(a=ureg.m, b=ureg.m),
        pass_as_kwargs=("a", "b"),
    )
    def add_together(a, b):
        return a + b

    add_together(a=ureg.Quantity(7.0, ureg.m), b=ureg.Quantity(3.0, ureg.feet))


@pytest.mark.filterwarnings("error")
def test_pass_as_kwargs_in_wrong_order():
    with pytest.raises(ValueError):

        @wraps_ufunc(
            return_units=dict(result=ureg.m),
            input_units=dict(a=ureg.m, b=ureg.m),
            pass_as_kwargs=("a"),
        )
        def add_together(a, b):
            return a + b

        add_together(a=ureg.Quantity(7.0, ureg.m), b=ureg.Quantity(3.0, ureg.feet))


@pytest.mark.filterwarnings("error")
def test_multiple_return():
    @wraps_ufunc(
        return_units=dict(b=ureg.m, a=ureg.m),
        input_units=dict(a=ureg.m, b=ureg.m),
        output_core_dims=[(), ()],
    )
    def swap(a, b):
        return b, a

    a = xr.DataArray(ureg.Quantity(np.linspace(0.0, 100.0), ureg.m))
    b = ureg.Quantity(np.pi, ureg.m)

    b2, a2 = swap(a, b)

    assert check_equal(a2, a)
    assert np.all(np.abs(b2 - b) < ureg.Quantity(1.0, ureg.mm))

    assert isinstance(a2, xr.DataArray)
    assert isinstance(b2, xr.DataArray)

    assert a2.pint.units == ureg.m
    assert b2.pint.units == ureg.m


@pytest.mark.filterwarnings("error")
def test_multiple_return_with_wrong_number_of_units():
    @wraps_ufunc(
        return_units=dict(a=ureg.m),
        input_units=dict(a=ureg.m, b=ureg.m),
        output_core_dims=[(), ()],
    )
    def swap(a, b):
        return b, a

    a = xr.DataArray(ureg.Quantity(np.linspace(0.0, 100.0), ureg.m))
    b = ureg.Quantity(np.pi, ureg.m)

    with pytest.raises(ValueError):
        b2, a2 = swap(a, b)


@pytest.mark.filterwarnings("error")
def test_no_return():
    @wraps_ufunc(return_units=dict(), input_units=dict(a=ureg.m))
    def do_nothing(a):
        pass

    a = ureg.Quantity(np.pi, ureg.m)
    do_nothing(a)


@pytest.mark.filterwarnings("error")
def test_no_return_with_too_many_units():
    @wraps_ufunc(return_units=dict(a=ureg.m), input_units=dict(a=ureg.m))
    def do_nothing(a):
        pass

    a = ureg.Quantity(np.pi, ureg.m)

    # Returns Quantity(None, ureg.m)
    do_nothing(a)


@pytest.mark.filterwarnings("error")
def test_illegal_chained_call_input():
    @wraps_ufunc(return_units=dict(doubled=None), input_units=dict(x=ureg.m))
    def simple_function(x):
        return 2 * x

    @wraps_ufunc(return_units=dict(doubled=None), input_units=dict(x=ureg.m))
    def simple_function2(x):
        return simple_function(x)

    with pytest.raises(
        RuntimeError,
        match=r".*Calling `wraps_ufunc` decorated function from within.*\n.*\n.*\n.*simple_function.unitless_func.*",
    ):
        simple_function2(10 * ureg.m)


@pytest.mark.filterwarnings("error")
def test_illegal_chained_call_ouput():
    @wraps_ufunc(return_units=dict(doubled=ureg.m), input_units=dict(x=None))
    def simple_function(x):
        return 2 * x

    @wraps_ufunc(return_units=dict(doubled=ureg.m), input_units=dict(x=None))
    def simple_function2(x):
        return simple_function(x)

    with pytest.raises(UnitStrippedWarning):
        simple_function2(10)
