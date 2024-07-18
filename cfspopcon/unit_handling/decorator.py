"""Defines the wraps_ufunc decorator used to perform unit conversions and dimension handling."""

import functools
import warnings
from collections.abc import Callable, Mapping, Sequence, Set
from inspect import Parameter, Signature, signature
from types import GenericAlias
from typing import Any, Optional, Union

import xarray as xr
from pint import Unit, UnitStrippedWarning

from .setup_unit_handling import Quantity, convert_units, magnitude, ureg

FunctionType = Callable[..., Any]


def wraps_ufunc(  # noqa: PLR0915
    input_units: dict[str, Union[str, Unit, None]],
    return_units: dict[str, Union[str, Unit, None]],
    pass_as_kwargs: tuple = (),
    #  kwargs for apply_ufunc
    input_core_dims: Optional[Sequence[Sequence]] = None,
    output_core_dims: Optional[Sequence[Sequence]] = ((),),
    exclude_dims: Set = frozenset(),
    vectorize: bool = True,
    join: str = "exact",
    dataset_join: str = "exact",
    keep_attrs: str = "drop_conflicts",
    dask: str = "forbidden",
    output_dtypes: Optional[Sequence] = None,
    output_sizes: Optional[Mapping[Any, int]] = None,
    dask_gufunc_kwargs: Optional[dict[str, Any]] = None,
) -> FunctionType:
    """Decorator for functions to add in unit and dimension handling.

    input_units and return_units must be provided, as dictionaries giving
    a mapping between the function arguments/returns and their units.

    pass_as_kwargs can be used to optionally declare that specific arguments
    should be pass directly into the function, rather than vectorized.

    The remaining arguments for the wrapper correspond to arguments for
    xr.apply_ufunc.
    https://docs.xarray.dev/en/stable/examples/apply_ufunc_vectorize_1d.html
    """
    input_units = _check_units(input_units)
    return_units = _check_units(return_units)

    ufunc_kwargs: dict[str, Any] = dict(
        input_core_dims=input_core_dims,
        output_core_dims=output_core_dims,
        exclude_dims=exclude_dims,
        vectorize=vectorize,
        join=join,
        dataset_join=dataset_join,
        keep_attrs=keep_attrs,
        dask=dask,
        output_dtypes=output_dtypes,
        output_sizes=output_sizes,
        dask_gufunc_kwargs=dask_gufunc_kwargs,
    )
    input_keys = list(input_units.keys())

    if not isinstance(pass_as_kwargs, tuple):
        raise ValueError(f"pass_as_kwargs must be passed as a tuple of keys, not {str(type(pass_as_kwargs))[1:-1]}")

    pass_as_positional_args = [key for key in input_keys if key not in pass_as_kwargs]
    for arg in pass_as_kwargs:
        kwarg_position = input_keys.index(arg)
        if kwarg_position < len(pass_as_positional_args):
            raise ValueError(f"Argument {arg} in pass_as_kwargs appears before the positional args {pass_as_positional_args}")

    if input_core_dims is not None:
        if not len(input_core_dims) == len(pass_as_positional_args):
            raise ValueError(
                f"input_core_dims (len {len(input_core_dims)}) must the same length as positional_args ({pass_as_positional_args}, len {len(pass_as_positional_args)})"
            )
    else:
        input_core_dims = len(pass_as_positional_args) * [()]

    def _wraps_ufunc(func: FunctionType) -> FunctionType:
        func_signature = signature(func)
        func_parameters = func_signature.parameters

        if not list(input_units.keys()) == list(func_parameters.keys()):
            raise ValueError(
                f"Keys for input_units {input_units.keys()} did not match func_parameters {func_parameters.keys()} (n.b. order matters!)"
            )

        default_values = {key: val.default for key, val in func_parameters.items() if val.default is not Parameter.empty}

        @functools.wraps(func)
        def popcon_ufunc_wrapped_call(*args: Any, **kwargs: Any) -> Any:  # noqa: PLR0912
            """Transform args and kwargs, then call the inner function."""
            # if anything goes wrong we can do some extra work to provide a better error below
            try:
                args_dict = dict(zip(input_keys, args))

                if not set(args_dict.keys()).isdisjoint(kwargs.keys()):
                    raise RuntimeError(
                        f"{func.__name__} was called with repeat arguments. Input was interpreted as args={args_dict}, kwargs={kwargs}"
                    )

                args_dict = {**args_dict, **kwargs}
                args_dict = {**args_dict, **{key: val for key, val in default_values.items() if key not in args_dict.keys()}}

                args_dict = _return_magnitude_in_specified_units(args_dict, input_units)

                positional_args = []
                for i, key in enumerate(pass_as_positional_args):
                    arg = args_dict[key]
                    if not isinstance(arg, xr.DataArray):
                        positional_args.append(xr.DataArray(arg).expand_dims(input_core_dims[i]))
                    else:
                        positional_args.append(arg)

                with warnings.catch_warnings():
                    warnings.simplefilter("error", category=UnitStrippedWarning)
                    function_return = xr.apply_ufunc(
                        func,
                        *positional_args,
                        kwargs={key: args_dict[key] for key in pass_as_kwargs},
                        **ufunc_kwargs,
                    )

                if len(return_units) == 0:
                    # Assume that the function return None
                    return function_return.item()

                function_return = _convert_return_to_quantities(function_return, return_units)

                function_return = list(function_return.values())

                if len(function_return) > 1:
                    return tuple(function_return)
                else:
                    return function_return[0]

            except Exception as e:
                # the below checks if we are inside FunctionWrapper being called from another FunctionWrapper
                # if that is the case we try and give a more helpful error
                # if anything goes wrong in our frame inspection or we find that we aren't in a chained
                # call we raise the previous exception
                err = ""
                try:
                    import inspect

                    frames = inspect.getouterframes(inspect.currentframe())
                    # the first entry is the current call so check if any of the earlier callees are a __call__ from a FunctionWrapper
                    for frame in frames[1:]:
                        if frame.function == "popcon_ufunc_wrapped_call":
                            f = frames[1]
                            err = "Calling `wraps_ufunc` decorated function from within `wraps_ufunc` decorated function is not allowed!\n"
                            err += f"Error at {f.filename}:{f.lineno}\n"
                            err += "\n".join(f.code_context) if f.code_context else ""
                            err += f"Try using `{frames[0].frame.f_locals['func'].__name__}.unitless_func(...)` instead."
                            break
                except Exception:
                    # error while determining if we are withing a chained FunctionWrapper so re-raise original error
                    raise e from None

                # if err is not empty we have determined we are within a chained call so we raise a better error
                if err:
                    raise RuntimeError(err) from None
                else:
                    raise e

        # more meaningfull alias to the scalar non-unit version of the function
        popcon_ufunc_wrapped_call.unitless_func = popcon_ufunc_wrapped_call.__wrapped__  # type:ignore[attr-defined]
        popcon_ufunc_wrapped_call.__signature__ = _make_new_sig(func_signature, input_units, return_units)  # type:ignore[attr-defined]
        return popcon_ufunc_wrapped_call

    return _wraps_ufunc


def _check_units(units_dict: dict[str, Union[str, Unit, None]]) -> dict[str, Union[str, Unit, None]]:
    for key, unit in units_dict.items():
        if unit is None:
            pass
        elif isinstance(unit, str):
            units_dict[key] = ureg(unit).units
        elif not isinstance(unit, Unit):
            raise TypeError(f"wraps_ufunc units for {key} must by of type str or Unit, not {str(type(unit))[1:-1]} (value was {unit})")

    return units_dict


def _return_magnitude_in_specified_units(vals: Any, units_mapping: dict[str, Union[str, Unit, None]]) -> dict[str, Any]:
    if not set(vals.keys()) == set(units_mapping):
        raise ValueError(f"Argument keys {vals.keys()} did not match units_mapping keys {units_mapping.keys()}")

    converted_vals = {}

    for key in vals.keys():
        val = vals[key]
        unit = units_mapping[key]

        if unit is None or val is None:
            converted_vals[key] = val

        elif isinstance(val, Quantity):
            converted_vals[key] = magnitude(convert_units(val, unit))

        elif isinstance(val, xr.DataArray):
            converted_vals[key] = convert_units(val, unit).pint.dequantify()

        elif Quantity(1, unit).check(ureg.dimensionless):
            converted_vals[key] = val

        else:
            raise NotImplementedError(f"Cannot convert {key} of type {str(type(val))[1:-1]} to units {unit}")

    return converted_vals


def _convert_return_to_quantities(vals: Any, units_mapping: dict[str, Union[str, Unit, None]]) -> dict[str, Any]:
    if not isinstance(vals, tuple):
        vals = (vals,)

    if not len(vals) == len(units_mapping):
        raise ValueError(f"Number of returned values ({len(vals)}) did not match length of units_mapping ({len(units_mapping)})")
    vals = dict(zip(units_mapping.keys(), vals))

    converted_vals = {}

    for key in vals.keys():
        val = vals[key]
        unit = units_mapping[key]

        if unit is None or val is None:
            converted_vals[key] = val

        elif isinstance(val, xr.DataArray):
            converted_vals[key] = val.pint.quantify(unit, unit_registry=ureg)

        elif isinstance(val, Quantity):
            converted_vals[key] = val.to(unit)

        else:
            converted_vals[key] = Quantity(val, unit)

    return converted_vals


def _make_new_sig(
    sig: Signature,
    input_units: Mapping[str, Union[str, Unit, None]],
    return_units: Mapping[str, Union[str, Unit, None]],
) -> Signature:
    """Create a new signature for a wrapped function that replaces the plain floats/arrays with Quantity/DataArray."""
    parameters = list(sig.parameters.values())
    ret_annotation = sig.return_annotation

    # update parameter annotations
    new_parameters: list[Parameter] = []
    for param, unit in zip(parameters, input_units.values()):
        if unit is None:
            new_parameters.append(param)
        else:
            new_parameters.append(param.replace(annotation=Union[Quantity, xr.DataArray]))

    # update return annotation
    units_list = list(return_units.values())

    # extract the types from the tuple
    if isinstance(ret_annotation, GenericAlias) and ret_annotation.__origin__ is tuple:
        old_types: list[Any] = list(ret_annotation.__args__)
    elif ret_annotation == Parameter.empty:
        old_types = [Any for _ in range(len(units_list))]
    else:
        old_types = [ret_annotation]

    if len(old_types) != len(units_list):
        if not (
            # Catches an error where some multiple-return types are handled as strings. These can
            # be safely ignored.
            isinstance(ret_annotation, str)
            and ret_annotation.startswith("tuple")
            and len(ret_annotation.removeprefix("tuple[").removesuffix("]").split(",")) == len(units_list)
        ):
            warnings.warn(
                (
                    f"Return type annotation {ret_annotation} has {len(old_types)} return values"
                    f", while the return_units: {return_units} specifies {len(return_units)} values"
                ),
                stacklevel=3,
            )

    ret_types = tuple(xr.DataArray if units_list[i] is not None else old_types[i] for i in range(len(units_list)))

    if len(ret_types) == 0:
        new_ret_ann: Union[type, None, GenericAlias] = None
    elif len(ret_types) == 1:
        new_ret_ann = ret_types[0]
    else:
        new_ret_ann = GenericAlias(tuple, ret_types)

    return sig.replace(parameters=new_parameters, return_annotation=new_ret_ann)
