"""Defines a class for different POPCON algorithms."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from functools import wraps
from pathlib import Path  # noqa: TCH003
from typing import Any, ClassVar, Optional, Union
from warnings import warn

import xarray as xr
import yaml

from .unit_handling import Quantity, convert_to_default_units, ureg

LabelledReturnFunctionType = Callable[..., dict[str, Any]]
GenericFunctionType = Callable[..., Any]


class Algorithm:
    """A class which handles the input and output of POPCON algorithms."""

    instances: ClassVar[dict[str, Algorithm]] = dict()

    def __init__(
        self,
        function: LabelledReturnFunctionType,
        return_keys: list[str],
        name: Optional[str] = None,
        skip_registration: bool = False,
    ):
        """Initialise an Algorithm.

        Args:
            function: a callable function
            return_keys: the arguments which are returned from the function
            name: Descriptive name for algorithm
            skip_registration: flag to skip adding the Algorithm to 'instances' (useful for testing)
        """
        self._function = function
        self._name = self._function.__name__ if name is None else name
        key = self._name.removeprefix("run_")
        if key in self.instances:
            raise RuntimeError(f"Algorithm {key} has been defined multiple times.")
        if not skip_registration:
            self.instances[key] = self

        self._signature = inspect.signature(function)
        for p in self._signature.parameters.values():
            if p.kind not in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
                inspect.Parameter.VAR_KEYWORD,
            ):
                raise ValueError(
                    f"Algorithm only supports functions with keyword arguments, but {function}, has {p.kind} parameter {p.name}"
                )
        self.input_keys = list(self._signature.parameters.keys())
        self.return_keys = return_keys

        self.default_values = {
            key: val.default
            for key, val in self._signature.parameters.items()
            if val.default is not inspect.Parameter.empty
        }
        self.default_keys = list(self.default_values.keys())

        self.required_input_keys = [
            key for key in self.input_keys if key not in self.default_keys
        ]

        self.__doc__ = self._make_docstring()

        self.run = self._make_run(self._function)

    def _make_docstring(self) -> str:
        """Makes a doc-string detailing the function inputs and outputs."""
        return_string = (
            f"Algorithm: {self._name}\n"
            + "Inputs:\n"
            + ", ".join(self.input_keys)
            + "\n"
            + "Outputs:\n"
            + ", ".join(self.return_keys)
        )
        return return_string

    def __repr__(self) -> str:
        """Return a simple string description of the Algorithm."""
        return f"Algorithm: {self._name}"

    @classmethod
    def _make_run(cls, func: LabelledReturnFunctionType) -> Callable[..., xr.Dataset]:
        """Helper to create the `run()` function with correct doc string.

        Args:
            func: function to be wrapped

        Returns: a xarray DataSet of the result
        """

        @wraps(func)
        def run(**kwargs: Any) -> xr.Dataset:
            result = func(**kwargs)
            dataset = xr.Dataset(result)
            return dataset

        return run

    def update_dataset(
        self, dataset: xr.Dataset, allow_overwrite: bool = True
    ) -> xr.Dataset:
        """Retrieve inputs from passed dataset and return a new dataset combining input and output quantities.

        Args:
            dataset: input dataset
            allow_overwrite: if False, raise an error if trying to write a variable which is already defined in dataset

        Returns: modified dataset
        """
        input_values = {}
        for key in self.input_keys:
            if key in dataset.keys():
                input_values[key] = dataset[key]
            elif key in self.default_keys:
                input_values[key] = self.default_values[key]
            else:
                sorted_dataset_keys = ", ".join(
                    sorted(dataset.keys())
                )  # type:ignore[arg-type]
                sorted_default_keys = ", ".join(sorted(self.default_keys))
                raise KeyError(
                    f"KeyError for {self._name}: Key '{key}' not in dataset keys [{sorted_dataset_keys}] or default values [{sorted_default_keys}]"
                )

        result = self._function(**input_values)
        return xr.Dataset(result).merge(
            dataset,
            join="left",
            compat=("override" if allow_overwrite else "no_conflicts"),
        )

    def __add__(
        self, other: Union[Algorithm, CompositeAlgorithm]
    ) -> CompositeAlgorithm:
        """Build a CompositeAlgorithm composed of this Algorithm and another Algorithm or CompositeAlgorithm."""
        if isinstance(other, CompositeAlgorithm):
            return CompositeAlgorithm(algorithms=[self, *other.algorithms])
        else:
            return CompositeAlgorithm(algorithms=[self, other])

    @classmethod
    def from_single_function(
        cls,
        func: Callable,
        return_keys: list[str],
        name: Optional[str] = None,
        skip_unit_conversion: bool = False,
        skip_registration: bool = False,
    ) -> Algorithm:
        """Build an Algorithm which wraps a single function."""

        @wraps(func)
        def wrapped_function(**kwargs: Any) -> dict:
            result = func(**kwargs)

            if not isinstance(result, tuple):
                result = (result,)

            result_dict = {}
            for i, key in enumerate(return_keys):
                if skip_unit_conversion:
                    result_dict[key] = result[i]
                elif isinstance(result[i], float):
                    result_dict[key] = convert_to_default_units(
                        Quantity(result[i], ureg.dimensionless), key
                    )
                else:
                    result_dict[key] = convert_to_default_units(result[i], key)

            return result_dict

        return cls(
            wrapped_function,
            return_keys,
            name=name if name is not None else func.__name__,
            skip_registration=skip_registration,
        )

    @classmethod
    def register_algorithm(
        cls,
        return_keys: list[str],
        name: Optional[str] = None,
        skip_unit_conversion: bool = False,
    ) -> GenericFunctionType:
        """Decorate a function and turn it into an Algorithm. Usage: @Algorithm.register_algorithm(return_keys=["..."])."""  # noqa: D402

        def function_wrapper(func: GenericFunctionType) -> GenericFunctionType:
            Algorithm.from_single_function(
                func,
                return_keys=return_keys,
                name=name if name is not None else func.__name__,
                skip_unit_conversion=skip_unit_conversion,
            )
            return func

        return function_wrapper

    def validate_inputs(
        self,
        configuration: Union[dict, xr.Dataset],
        quiet: bool = False,
        raise_error_on_missing_inputs: bool = False,
    ) -> bool:
        """Check that all required inputs are defined, and warn if inputs are unused."""
        return _validate_inputs(
            self,
            configuration,
            quiet=quiet,
            raise_error_on_missing_inputs=raise_error_on_missing_inputs,
        )

    @classmethod
    def write_yaml(cls, filepath: Path) -> None:
        """Writes a file 'algorithms.yaml' documenting the available algorithms."""
        data = dict()

        for name, alg in cls.instances.items():
            alg_data = dict()
            alg_data["inputs"] = alg.required_input_keys
            alg_data["optionals"] = alg.default_keys
            alg_data["returns"] = alg.return_keys

            data[name] = alg_data

        yaml_text = yaml.dump(dict(sorted(data.items())))

        with open(filepath, "w") as f:
            f.write("# Autogenerated by Algorithm.write_yaml()\n\n")
            f.write(yaml_text)

    @classmethod
    def algorithms(cls) -> list[str]:
        """Make a list of the available algorithms."""
        return list(cls.instances.keys())

    @classmethod
    def get_algorithm(cls, key: str) -> Algorithm:
        """Retrieves an algorithm by name."""
        if key not in cls.algorithms():
            error_message = (
                f"algorithm '{key}' not found. "
                "If you have constructed or registered an Algorithm of this name, "
                "make sure that it is imported in the top-level cfspopcon __init__.py. "
                "Algorithms which have been successfully registered and imported will "
                "appear in the algorithms.yaml file."
            )
            raise KeyError(error_message)

        return cls.instances[key]


class CompositeAlgorithm:
    """A class which combined multiple Algorithms into a single object which behaves like an Algorithm."""

    def __init__(
        self,
        algorithms: Sequence[Union[Algorithm, CompositeAlgorithm]],
        name: Optional[str] = None,
    ):
        """Initialise a CompositeAlgorithm, combining several other Algorithms.

        Args:
            algorithms: a list of Algorithms, in the order that they should be executed.
            name: a name used to refer to the composite algorithm.
        """
        if not (
            isinstance(algorithms, Sequence)
            and all(
                isinstance(alg, (Algorithm, CompositeAlgorithm)) for alg in algorithms
            )
        ):
            raise TypeError(
                "Should pass a list of algorithms or composites to CompositeAlgorithm."
            )

        self.algorithms: list[Algorithm] = []

        # flattens composite algorithms into their respective list of plain Algorithms
        for alg in algorithms:
            if isinstance(alg, Algorithm):
                self.algorithms.append(alg)
            else:
                self.algorithms.extend(alg.algorithms)

        self.input_keys: list[str] = []
        self.required_input_keys: list[str] = []
        self.return_keys: list[str] = []
        pars: list[inspect.Parameter] = []

        # traverse list of algorithms in order.
        # If an ouput from the set of previous algorithms provides an input to a following algorithm
        # the input is not turned into an input to the CompositeAlgorithm
        for alg in self.algorithms:
            alg_sig = inspect.signature(alg.run)
            for key in alg.default_keys:
                if key not in self.return_keys:
                    self.input_keys.append(key)
                    pars.append(alg_sig.parameters[key])
            for key in alg.required_input_keys:
                if key not in self.return_keys:
                    self.input_keys.append(key)
                    self.required_input_keys.append(key)
                    pars.append(alg_sig.parameters[key])

            for key in alg.return_keys:
                if key not in self.return_keys:
                    self.return_keys.append(key)

        # create a signature for the run() function
        # This is a purely aesthetic change, that ensures the run() function
        # has a helpful tooltip in editors and in the documentation

        # 1. make sure the list of pars doesn't have any duplicates, if there are duplicates
        # we pick the first one. We don't assert that the types of two parameters are compatible
        # that's not easy to do.
        seen_pars: dict[str, int] = {}
        pars = [p for i, p in enumerate(pars) if seen_pars.setdefault(p.name, i) == i]

        # ensure POSITIONAL_OR_KEYWORD are before kw only
        pars = sorted(pars, key=lambda p: p.kind)

        def_pars = [p for p in pars if p.default != inspect.Parameter.empty]
        non_def_pars = [p for p in pars if p.default == inspect.Parameter.empty]

        # methods are immutable and we don't want to set a signature on the class' run() method
        # thus we wrap the original run method and then assign the __signature__ to the wrapped
        # wrapper function
        def _wrap(f: Callable[..., xr.Dataset]) -> Callable[..., xr.Dataset]:
            def wrapper(**kwargs: Any) -> xr.Dataset:
                return f(**kwargs)

            wrapper.__doc__ = f.__doc__

            return wrapper

        self.run = _wrap(self._run)
        # ignore due to mypy bug/missing feature https://github.com/python/mypy/issues/3482
        self.run.__signature__ = inspect.Signature(  # type:ignore[attr-defined]
            non_def_pars + def_pars, return_annotation=xr.Dataset
        )
        self._name = name
        self.__doc__ = self._make_docstring()

    def _make_docstring(self) -> str:
        """Makes a doc-string detailing the function inputs and outputs."""
        components = f"[{', '.join(alg._name for alg in self.algorithms)}]"

        return_string = (
            f"CompositeAlgorithm: {self._name}\n"
            if self._name is not None
            else "CompositeAlgorithm\n"
            f"Composed of {components}\n"
            f"Inputs:\n{', '.join(self.input_keys)}\n"
            f"Outputs:\n{', '.join(self.return_keys)}"
        )
        return return_string

    def __repr__(self) -> str:
        """Return a simple string description of the CompositeAlgorithm."""
        return f"CompositeAlgorithm: {self._name}"

    def _run(self, **kwargs: Any) -> xr.Dataset:
        """Run the sub-Algorithms, one after the other and return a xarray.Dataset of the results.

        Will throw a warning if parameters are not used by any sub-Algorithm.
        """
        result = kwargs

        parameters_extra = set(kwargs) - set(self.required_input_keys)
        parameters_missing = set(self.required_input_keys) - set(kwargs)
        if parameters_missing:
            needed_by: dict[str, list] = dict()

            for parameter in parameters_missing:
                needed_by[parameter] = []
                for alg in self.algorithms:
                    if parameter in alg.input_keys:
                        needed_by[parameter].append(alg._name)

            error_string = ", ".join(
                f"{key} needed by [{', '.join(val)}]" for key, val in needed_by.items()
            )
            raise TypeError(
                f"CompositeAlgorithm.run() missing arguments: {error_string}"
            )
        if parameters_extra:
            warn(
                f"Not all input parameters were used. Unused parameters: [{', '.join(parameters_extra)}]",
                stacklevel=3,
            )

        for alg in self.algorithms:
            alg_kwargs = {
                key: result[key] for key in result.keys() if key in alg.input_keys
            }

            alg_result = alg.run(**alg_kwargs)
            result.update(
                alg_result
            )  # type:ignore[arg-type]  # dict.update() doesn't like KeysView[Hashable]

        return xr.Dataset(result)

    def update_dataset(
        self, dataset: xr.Dataset, allow_overwrite: bool = True
    ) -> xr.Dataset:
        """Retrieve inputs from passed dataset and return a new dataset combining input and output quantities.

        N.b. will not throw a warning if the dataset contains unused elements.

        Args:
            dataset: input dataset
            allow_overwrite: if False, raise an error if trying to write a variable which is already defined in dataset

        Returns: modified dataset
        """
        for alg in self.algorithms:
            dataset = alg.update_dataset(dataset, allow_overwrite=allow_overwrite)

        return dataset

    def __add__(
        self, other: Union[Algorithm, CompositeAlgorithm]
    ) -> CompositeAlgorithm:
        """Build a CompositeAlgorithm composed of this CompositeAlgorithm and another Algorithm or CompositeAlgorithm."""
        if isinstance(other, Algorithm):
            return CompositeAlgorithm(algorithms=[*self.algorithms, other])
        else:
            return CompositeAlgorithm(algorithms=[*self.algorithms, *other.algorithms])

    def validate_inputs(  # noqa: PLR0912
        self,
        configuration: Union[dict, xr.Dataset],
        quiet: bool = False,
        raise_error_on_missing_inputs: bool = True,
        warn_for_overridden_variables: bool = False,
    ) -> bool:
        """Check that all required inputs are defined, and warn if inputs are unused."""
        # Check if variables are being silently internally overwritten
        config_keys = list(configuration.keys())
        key_setter = {key: ["INPUT"] for key in config_keys}

        for algorithm in self.algorithms:
            for key in algorithm.return_keys:
                if key not in key_setter.keys():
                    key_setter[key] = [algorithm._name]
                else:
                    key_setter[key].append(algorithm._name)

        overridden_variables = []
        for variable, algs in key_setter.items():
            if len(algs) > 1:
                overridden_variables.append(f"{variable}: ({', '.join(algs)})")

        if warn_for_overridden_variables and len(overridden_variables) > 0:
            warn(
                f"The following variables were overridden internally (given as variable: (list of algorithms setting variable)): {', '.join(overridden_variables)}",
                stacklevel=3,
            )

        # Check that algorithms are ordered such that dependent algorithms follow those setting their required input keys
        available_parameters = config_keys.copy()
        out_of_order_parameters = {}
        for algorithm in self.algorithms:
            for key in algorithm.required_input_keys:
                if key not in available_parameters:
                    out_of_order_parameters[key] = algorithm
            for key in algorithm.return_keys:
                available_parameters.append(key)

        if len(out_of_order_parameters) > 0:
            message = ""
            for key, algorithm in out_of_order_parameters.items():
                if key in key_setter and len(key_setter.get(key, [])) > 0:
                    message += f"{key} needed by {algorithm} defined by output of {key_setter[key]}."
            if len(message) > 0:
                message = f"Algorithms out of order. {message}. Rearrange the list of algorithms so that dependent algorithm are after algorithms setting their inputs."
                if raise_error_on_missing_inputs:
                    raise RuntimeError(message)
                if not quiet:
                    warn(message, stacklevel=3)

            _validate_inputs(
                self,
                configuration,
                quiet=quiet,
                raise_error_on_missing_inputs=raise_error_on_missing_inputs,
            )

            return False
        else:
            return _validate_inputs(
                self,
                configuration,
                quiet=quiet,
                raise_error_on_missing_inputs=raise_error_on_missing_inputs,
            )


def _validate_inputs(
    algorithm: Union[Algorithm, CompositeAlgorithm],
    configuration: Union[dict, xr.Dataset],
    quiet: bool = False,
    raise_error_on_missing_inputs: bool = False,
) -> bool:
    """Check that all required inputs are defined, and warn if inputs are unused."""
    config_keys = list(configuration.keys())

    unused_config_keys = config_keys.copy()
    missing_input_keys = set(algorithm.required_input_keys)

    for key in config_keys:
        if key in missing_input_keys:
            missing_input_keys.remove(key)

        if key in algorithm.input_keys:
            # required_input_keys gives the list of keys which must
            # be provided, while input_puts gives the list of keys
            # which can be provided (but which might have default values).
            unused_config_keys.remove(key)

    if len(missing_input_keys) == 0 and len(unused_config_keys) == 0:
        return True

    elif len(missing_input_keys) > 0 and len(unused_config_keys) > 0:
        message = f"Missing input parameters [{', '.join(missing_input_keys)}]. Also had unused input parameters [{', '.join(unused_config_keys)}]."
        if raise_error_on_missing_inputs:
            raise RuntimeError(message)

    elif len(missing_input_keys) > 0:
        message = f"Missing input parameters [{', '.join(missing_input_keys)}]."
        if raise_error_on_missing_inputs:
            raise RuntimeError(message)

    else:
        message = f"Unused input parameters [{', '.join(unused_config_keys)}]."

    if not quiet:
        warn(message, stacklevel=3)
    return False
