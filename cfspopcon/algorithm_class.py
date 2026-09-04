"""Algorithms, composites, and the registry which holds them.

A plugin is registered deliberately, by its import name::

    import cfspopcon

    cfspopcon.register_plugin("my_popcon_plugin")

If you did not register the bundled plugin (``cfspopcon.formulas``), the first use of the
registry will trigger it automatically::

    # registers the bundled plugin
    volume_algorithm = cfspopcon.registry["calc_plasma_volume"]
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import sys
from collections.abc import Callable, Iterator, Sequence
from difflib import get_close_matches
from functools import wraps
from importlib.resources import files
from pathlib import Path  # noqa: TC003
from typing import Any, ClassVar
from warnings import warn

import xarray as xr
import yaml

from .unit_handling import Quantity, convert_to_default_units, ureg
from .unit_handling.default_units import (
    default_units_map,
    extend_default_units_map,
    read_default_units_from_file,
    reset_default_units,
)

# A function whose outputs come back as a {return key: value} mapping.
LabelledReturnFunctionType = Callable[..., dict[str, Any]]

# A function with any signature and any return shape.
GenericFunctionType = Callable[..., Any]

#: The bundled plugin: the package holding cfspopcon's own algorithms.
_BUNDLED_PLUGIN = "cfspopcon.formulas"

#: Whether registration of the bundled algorithms has started.
_BUNDLED_ALGORITHMS_DISCOVERED = False

#: Packages whose registration is on the call stack, guarding against circular requirements.
_REGISTRATION_IN_PROGRESS: set[str] = set()


def _algorithm_not_found_message(key: str) -> str:
    """Explain as specifically as possible why an algorithm name did not resolve."""
    close_matches = get_close_matches(key, list(Algorithm.instances), n=1)
    if close_matches:
        return f"algorithm '{key}' not found. Did you mean '{close_matches[0]}'?"

    return (
        f"algorithm '{key}' not found. If it comes from a plugin, register the plugin first: list it "
        "in the input file's plugins section, or call register_plugin. "
        "Run popcon_algorithms to list what is registered."
    )


def _register_algorithm(name: str, algorithm: Algorithm | CompositeAlgorithm, override: bool) -> None:
    """Add an algorithm to the registry, refusing to silently replace one of the same name."""
    if name in Algorithm.instances and not override:
        raise RuntimeError(f"Algorithm '{name}' is already registered. Pass override=True to replace it.")
    Algorithm.instances[name] = algorithm


class Algorithm:
    """A class which handles the input and output of POPCON algorithms."""

    #: The registered algorithms, keyed by name.
    instances: ClassVar[dict[str, Algorithm | CompositeAlgorithm]] = dict()

    def __init__(
        self,
        function: LabelledReturnFunctionType,
        return_keys: list[str],
        name: str | None = None,
        override: bool = False,
    ):
        """Initialise an Algorithm.

        Args:
            function: the function to wrap, taking keyword arguments and returning a
                ``{return key: value}`` mapping.
            return_keys: the variable names of the function's outputs, in the order they are
                returned.
            name: the algorithm's name. Defaults to the function's name.
            override: at registration, replace an already-registered algorithm of the same name.

        Raises:
            ValueError: if the function takes positional-only or variable positional arguments.
        """
        self._function = function
        self._name = self._function.__name__ if name is None else name
        self._override = override

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
            key: val.default for key, val in self._signature.parameters.items() if val.default is not inspect.Parameter.empty
        }
        self.default_keys = list(self.default_values.keys())

        self.required_input_keys = [key for key in self.input_keys if key not in self.default_keys]

        self.__doc__ = self._make_docstring()

        self.run = self._make_run(self._function)

    def _make_docstring(self) -> str:
        """Makes a doc-string detailing the function inputs and outputs."""
        return_string = (
            f"Algorithm: {self._name}\n" + "Inputs:\n" + ", ".join(self.input_keys) + "\n" + "Outputs:\n" + ", ".join(self.return_keys)
        )
        return return_string

    @property
    def name(self) -> str:
        """This algorithm's name, and its registry key once registered."""
        return self._name

    def __repr__(self) -> str:
        """Return a simple string description of the Algorithm."""
        return f"Algorithm: {self._name}"

    def __call__(self, *args: Any, return_labelled_dictionary: bool = False, **kwargs: Any) -> Any:
        """Call the algorithm like a function, returning its outputs directly.

        Example::

            algorithm = registry["algorithm_name"]
            outputs = algorithm(input_a=..., input_b=...)

        Args:
            *args: positional arguments for the algorithm's function.
            return_labelled_dictionary: return the outputs as a ``{return key: value}`` mapping.
            **kwargs: the algorithm's inputs.

        Returns:
            The output values in ``return_keys`` order (a single value for one output, a tuple
            for several), or the mapping if ``return_labelled_dictionary`` is set.
        """
        result_dict = self._function(*args, **kwargs)
        if return_labelled_dictionary:
            return result_dict
        else:
            results = tuple(result_dict[key] for key in self.return_keys)
            if len(results) == 1:
                return results[0]
            else:
                return results

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

    def update_dataset(self, dataset: xr.Dataset, allow_overwrite: bool = True) -> xr.Dataset:
        """Run the algorithm on the inputs found in a dataset, returning the dataset with the outputs merged in.

        Inputs are taken from the dataset by name, falling back to the algorithm's default values.

        Args:
            dataset: the dataset supplying the algorithm's inputs.
            allow_overwrite: whether an output may replace a variable already in the dataset.

        Returns:
            A new dataset combining the input dataset and the algorithm's outputs.

        Raises:
            KeyError: if a required input is in neither the dataset nor the default values.
        """
        input_values = {}
        for key in self.input_keys:
            if key in dataset.keys():
                input_values[key] = dataset[key]
            elif key in self.default_keys:
                input_values[key] = self.default_values[key]
            else:
                sorted_dataset_keys = ", ".join(sorted(dataset.keys()))  # type:ignore[arg-type]
                sorted_default_keys = ", ".join(sorted(self.default_keys))
                raise KeyError(
                    f"KeyError for {self._name}: Key '{key}' not in dataset keys [{sorted_dataset_keys}] or default values [{sorted_default_keys}]"
                )

        result = self._function(**input_values)
        return xr.Dataset(result).merge(dataset, join="left", compat=("override" if allow_overwrite else "no_conflicts"))

    def __add__(self, other: Algorithm | CompositeAlgorithm) -> CompositeAlgorithm:
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
        name: str | None = None,
        skip_unit_conversion: bool = False,
        override: bool = False,
    ) -> Algorithm:
        """Build an Algorithm from a function which returns its outputs plainly.

        Each output is normalized to its variable's default units.

        Args:
            func: the function to wrap, taking keyword arguments and returning one value per
                entry of ``return_keys``.
            return_keys: the variable names of the function's outputs, in the order they are
                returned.
            name: the algorithm's name. Defaults to the function's name.
            skip_unit_conversion: return the outputs as the function produces them, without
                normalizing each one to its variable's default units.
            override: at registration, replace an already-registered algorithm of the same name.

        Returns:
            The Algorithm wrapping the function.
        """
        if not isinstance(return_keys, list):
            return_keys = [return_keys]

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
                    result_dict[key] = convert_to_default_units(Quantity(result[i], ureg.dimensionless), key)
                else:
                    result_dict[key] = convert_to_default_units(result[i], key)

            return result_dict

        return cls(
            wrapped_function,
            return_keys,
            name=name if name is not None else func.__name__,
            override=override,
        )

    @classmethod
    def empty(cls) -> Algorithm:
        """Build an algorithm with no inputs and no outputs, for where an Algorithm is required but nothing should be computed.

        Returns:
            An Algorithm named ``"empty"`` which leaves a dataset unchanged.
        """

        def do_nothing() -> dict[str, Any]:
            result_dict: dict[str, Any] = {}
            return result_dict

        return cls(do_nothing, return_keys=[], name="empty")

    def validate_inputs(self, configuration: dict | xr.Dataset, quiet: bool = False, raise_error_on_missing_inputs: bool = False) -> bool:
        """Check a set of inputs against the algorithm's signature.

        Args:
            configuration: the inputs to check, as a mapping or dataset of variables.
            quiet: suppress the warning describing missing or unused inputs.
            raise_error_on_missing_inputs: raise instead of warning when required inputs are missing.

        Returns:
            True when every required input is present and every input is used.

        Raises:
            RuntimeError: if inputs are missing and ``raise_error_on_missing_inputs`` is set.
        """
        return _validate_inputs(self, configuration, quiet=quiet, raise_error_on_missing_inputs=raise_error_on_missing_inputs)


def algorithm(
    return_keys: list[str], name: str | None = None, skip_unit_conversion: bool = False, override: bool = False
) -> GenericFunctionType:
    """Label a function as an Algorithm, for :func:`register_plugin` to find.

    The algorithm enters the registry when the plugin defining the function is registered.

    Example::

        @algorithm(return_keys=["plasma_volume"])
        def calc_plasma_volume(major_radius, inverse_aspect_ratio, areal_elongation):
            ...

    Args:
        return_keys: the variable names of the function's outputs, in the order they are
            returned.
        name: the algorithm's name. Defaults to the function's name.
        skip_unit_conversion: return the outputs as the function produces them, without
            normalizing each one to its variable's default units.
        override: at registration, replace an already-registered algorithm of the same name.

    Returns:
        The decorator, which labels the function and returns it unchanged.
    """

    def function_wrapper(func: GenericFunctionType) -> GenericFunctionType:
        func.__popcon_algorithm__ = Algorithm.from_single_function(  # type:ignore[attr-defined]
            func,
            return_keys=return_keys,
            name=name if name is not None else func.__name__,
            skip_unit_conversion=skip_unit_conversion,
            override=override,
        )
        return func

    return function_wrapper


class CompositeAlgorithm:
    """A class which combined multiple Algorithms into a single object which behaves like an Algorithm."""

    def __init__(
        self,
        algorithms: Sequence[Algorithm | CompositeAlgorithm],
        name: str | None = None,
    ):
        """Initialise a CompositeAlgorithm, combining several other Algorithms.

        Args:
            algorithms: the Algorithms or CompositeAlgorithms to combine, in execution order.
            name: a name for the composite algorithm.

        Raises:
            TypeError: if ``algorithms`` is not a sequence of Algorithms or CompositeAlgorithms.
        """
        if not (isinstance(algorithms, Sequence) and all(isinstance(alg, Algorithm | CompositeAlgorithm) for alg in algorithms)):
            raise TypeError("Should pass a list of algorithms or composites to CompositeAlgorithm.")

        self.algorithms: list[Algorithm] = []

        # flattens composite algorithms into their respective list of plain Algorithms
        for alg in algorithms:
            if isinstance(alg, Algorithm):
                self.algorithms.append(alg)
            else:
                self.algorithms.extend(alg.algorithms)

        self.input_keys: list[str] = []
        self.default_keys: list[str] = []
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
                    self.default_keys.append(key)
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

    @classmethod
    def from_list(cls, keys: list[str], name: str | None = None) -> CompositeAlgorithm:
        """Build a CompositeAlgorithm from the names of registered algorithms.

        Args:
            keys: the names of the component algorithms, in execution order.
            name: a name for the composite algorithm.

        Returns:
            The composite of the named algorithms.
        """
        return cls([registry[key] for key in keys], name=name)

    @classmethod
    def declare(cls, keys: list[str], name: str, override: bool = False) -> CompositeDeclaration:
        """Declare a named CompositeAlgorithm, to be built once its components are registered.

        Component names are resolved at registration, so declarations may appear in any order. Assign the result at
        module level, ``my_chain = CompositeAlgorithm.declare([...], name="my_chain")``, and
        :func:`register_plugin` builds and registers it once the plugin's algorithms are in. Pass
        ``override=True`` to replace an already-registered algorithm of the composite's name. To
        build one now from already-registered algorithms, use :meth:`from_list`.

        Args:
            keys: the names of the component algorithms, in execution order.
            name: the name to register the built composite under.
            override: at registration, replace an already-registered algorithm of the same name.

        Returns:
            The declaration, to bind at module level.
        """
        return CompositeDeclaration(keys=list(keys), name=name, override=override)

    def _make_docstring(self) -> str:
        """Makes a doc-string detailing the function inputs and outputs."""
        components = f"[{', '.join(alg.name for alg in self.algorithms)}]"

        return_string = (f"CompositeAlgorithm: {self._name}\n" if self._name is not None else "CompositeAlgorithm\n") + (
            f"Composed of {components}\nInputs:\n{', '.join(self.input_keys)}\nOutputs:\n{', '.join(self.return_keys)}"
        )
        return return_string

    @property
    def name(self) -> str | None:
        """This composite's name (its registry key, once registered), or None if it is unnamed."""
        return self._name

    def __repr__(self) -> str:
        """Return a simple string description of the CompositeAlgorithm."""
        return f"CompositeAlgorithm: {self._name}"

    def _run(self, **kwargs: Any) -> xr.Dataset:
        """Run the component algorithms in order, returning a dataset of all inputs and outputs.

        Warns if an input is not used by any component, and raises a TypeError naming the
        algorithms which need a missing input.
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
                        needed_by[parameter].append(alg.name)

            error_string = ", ".join(f"{key} needed by [{', '.join(val)}]" for key, val in needed_by.items())
            raise TypeError(f"CompositeAlgorithm.run() missing arguments: {error_string}")
        if parameters_extra:
            warn(f"Not all input parameters were used. Unused parameters: [{', '.join(parameters_extra)}]", stacklevel=3)

        for alg in self.algorithms:
            alg_kwargs = {key: result[key] for key in result.keys() if key in alg.input_keys}

            alg_result = alg.run(**alg_kwargs)
            result.update(alg_result)  # type:ignore[arg-type]  # dict.update() doesn't like KeysView[Hashable]

        return xr.Dataset(result)

    def update_dataset(self, dataset: xr.Dataset, allow_overwrite: bool = True) -> xr.Dataset:
        """Run each component algorithm in turn on a dataset, returning the dataset with all outputs merged in.

        Unused dataset variables are passed through without a warning.

        Args:
            dataset: the dataset supplying the algorithms' inputs.
            allow_overwrite: whether an output may replace a variable already in the dataset.

        Returns:
            A new dataset combining the input dataset and every algorithm's outputs.
        """
        for alg in self.algorithms:
            dataset = alg.update_dataset(dataset, allow_overwrite=allow_overwrite)

        return dataset

    def __add__(self, other: Algorithm | CompositeAlgorithm) -> CompositeAlgorithm:
        """Build a CompositeAlgorithm composed of this CompositeAlgorithm and another Algorithm or CompositeAlgorithm."""
        if isinstance(other, Algorithm):
            return CompositeAlgorithm(algorithms=[*self.algorithms, other])
        else:
            return CompositeAlgorithm(algorithms=[*self.algorithms, *other.algorithms])

    def validate_inputs(  # noqa: PLR0912
        self,
        configuration: dict | xr.Dataset,
        quiet: bool = False,
        raise_error_on_missing_inputs: bool = True,
        warn_for_overridden_variables: bool = False,
    ) -> bool:
        """Check a set of inputs against the composite's signature and the ordering of its algorithms.

        Args:
            configuration: the inputs to check, as a mapping or dataset of variables.
            quiet: suppress the warning describing missing or unused inputs.
            raise_error_on_missing_inputs: raise instead of warning when required inputs are
                missing or the algorithms are out of order.
            warn_for_overridden_variables: warn when a variable is set by more than one algorithm.

        Returns:
            True when every required input is present, every input is used, and the algorithms
            are ordered so each one's inputs exist before it runs.

        Raises:
            RuntimeError: if ``raise_error_on_missing_inputs`` is set and inputs are missing or
                the algorithms are out of order.
        """
        # Check if variables are being silently internally overwritten
        config_keys = list(configuration.keys())
        key_setter = {key: ["INPUT"] for key in config_keys}

        for algorithm in self.algorithms:
            for key in algorithm.return_keys:
                if key not in key_setter.keys():
                    key_setter[key] = [algorithm.name]
                else:
                    key_setter[key].append(algorithm.name)

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

            _validate_inputs(self, configuration, quiet=quiet, raise_error_on_missing_inputs=raise_error_on_missing_inputs)

            return False
        else:
            return _validate_inputs(self, configuration, quiet=quiet, raise_error_on_missing_inputs=raise_error_on_missing_inputs)


def _validate_inputs(
    algorithm: Algorithm | CompositeAlgorithm,
    configuration: dict | xr.Dataset,
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

    missing = sorted(missing_input_keys)
    unused = sorted(unused_config_keys)

    if not missing and not unused:
        return True
    elif missing and unused:
        message = f"Missing input parameters [{', '.join(missing)}]. Also had unused input parameters [{', '.join(unused)}]."
    elif missing:
        message = f"Missing input parameters [{', '.join(missing)}]."
    else:
        message = f"Unused input parameters [{', '.join(unused)}]."
    message = "\n".join([message, *_input_hints(algorithm, missing, unused)])

    if missing and raise_error_on_missing_inputs:
        raise RuntimeError(message)
    if not quiet:
        warn(message, stacklevel=3)
    return False


def _input_hints(algorithm: Algorithm | CompositeAlgorithm, missing: list[str], unused: list[str]) -> list[str]:
    """Suggest a fix per missing or unused input, where the registry or a near-miss offers one."""
    hints = []
    for key in missing:
        setters = algorithms_setting(key)
        if setters:
            hints.append(f"'{key}' is set by [{', '.join(setters)}]: add one to your algorithms list, or provide '{key}' as an input.")
    for key in unused:
        close_matches = get_close_matches(key, algorithm.input_keys, n=1)
        if close_matches:
            hints.append(f"Unused parameter '{key}': did you mean '{close_matches[0]}'?")
    return hints


class CompositeDeclaration:
    """A named composite recorded as its component names, before the components exist.

    Registering the declaring plugin builds the composite once every component name resolves.
    """

    def __init__(self, keys: list[str], name: str, override: bool = False):
        """Record the component names, and the name to register the built composite under."""
        self.keys = keys
        self.name = name
        #: At registration, replace an already-registered algorithm of the same name.
        self.override = override
        #: The composite built from this declaration, once a registration has built it.
        self.built: CompositeAlgorithm | None = None


def _scan_plugin(plugin_name: str) -> tuple[list[Algorithm], list[CompositeDeclaration], list[str]]:
    """Collect the Algorithms, composite declarations, and requirements bound in a plugin's modules."""
    algorithms: list[Algorithm] = []
    declarations: list[CompositeDeclaration] = []
    requirements: list[str] = []
    seen: set[int] = set()
    for module_name in sorted(name for name in sys.modules if name == plugin_name or name.startswith(f"{plugin_name}.")):
        for attribute, value in list(vars(sys.modules[module_name]).items()):
            if attribute == "__popcon_requires__":
                if not isinstance(value, list | tuple) or not all(isinstance(entry, str) for entry in value):
                    raise ValueError(f"__popcon_requires__ in '{module_name}' must be a tuple of package names.")
                requirements.extend(value)
                continue
            candidate = getattr(value, "__popcon_algorithm__", value)
            if id(candidate) in seen:  # the same object may be re-exported by several modules
                continue
            if isinstance(candidate, Algorithm):
                seen.add(id(candidate))
                algorithms.append(candidate)
            elif isinstance(candidate, CompositeDeclaration):
                seen.add(id(candidate))
                declarations.append(candidate)
    return algorithms, declarations, requirements


def _register_scanned(algorithms: list[Algorithm], declarations: list[CompositeDeclaration]) -> None:
    """Register the scanned algorithms, then build and register the declared composites.

    A composite may be built from other composites, so each pass builds whichever declarations
    have all of their components registered, and repeats. A pass which can build nothing raises,
    naming the missing components. An already-registered object is skipped, so registering a
    plugin again changes nothing.
    """
    for algorithm in algorithms:
        if Algorithm.instances.get(algorithm.name) is algorithm:
            continue
        _register_algorithm(algorithm.name, algorithm, algorithm._override)

    pending = []
    for declaration in declarations:
        if declaration.built is not None:
            if Algorithm.instances.get(declaration.name) is not declaration.built:
                _register_algorithm(declaration.name, declaration.built, declaration.override)
            continue
        pending.append(declaration)

    while pending:
        ready = [d for d in pending if all(key in Algorithm.instances for key in d.keys)]
        if not ready:
            unresolved = "; ".join(
                f"'{d.name}' is missing [{', '.join(key for key in d.keys if key not in Algorithm.instances)}]" for d in pending
            )
            raise RuntimeError(f"Could not build the composite algorithms: {unresolved}.")
        for declaration in ready:
            built = CompositeAlgorithm([Algorithm.instances[key] for key in declaration.keys], name=declaration.name)
            _register_algorithm(declaration.name, built, override=declaration.override)
            declaration.built = built
            pending.remove(declaration)


def register_plugin(plugin_name: str) -> list[str]:
    """Register a plugin: its default units, its algorithms, and its composites.

    A ``variables.yaml`` in the package root is read into the default units map, every module
    beneath the package is imported (only directories containing an ``__init__.py`` are walked), and
    the Algorithms and composite declarations bound in those modules are then registered.
    The scan at the end of this call is what registers; the imports only build the objects, so an
    Algorithm a module imports from an unregistered package is registered as this plugin's. A
    composite may name anything registered by the end of its own plugin. The bundled algorithms
    are registered before any other plugin. Repeated calls change nothing.

    A plugin names the plugins whose registered algorithms it builds on with a module-level
    ``__popcon_requires__ = ("other_plugin",)``; each requirement is registered first, as its own
    registration.

    Registration is atomic: if anything fails, the registries are restored, and the plugin can
    be fixed and registered again in the same session. ``ureg.define`` calls are the exception;
    pint has no un-define.

    Args:
        plugin_name: the plugin's import name, e.g. ``"my_popcon_plugin"``, which may differ from
            the distribution name.

    Returns:
        The names of the algorithms this call added to the registry.

    Raises:
        RuntimeError: if a declared composite names a component which is not registered by the end
            of this plugin's registration, or the ``__popcon_requires__`` chain is circular.
        ValueError: if the package is a plain module, or its units change an existing variable's.
    """
    global _BUNDLED_ALGORITHMS_DISCOVERED  # noqa: PLW0603
    if plugin_name == _BUNDLED_PLUGIN:
        # Set before the walk: it doubles as the re-entrancy guard for registry use during it.
        _BUNDLED_ALGORITHMS_DISCOVERED = True
    elif not _BUNDLED_ALGORITHMS_DISCOVERED:
        register_plugin(_BUNDLED_PLUGIN)

    if plugin_name in _REGISTRATION_IN_PROGRESS:
        raise RuntimeError(f"Circular __popcon_requires__: '{plugin_name}' is already being registered.")
    _REGISTRATION_IN_PROGRESS.add(plugin_name)
    try:
        package = importlib.import_module(plugin_name)
        if not hasattr(package, "__path__"):
            raise ValueError(f"'{plugin_name}' is a plain module: a registration target must be a package.")
        # walk_packages only finds the modules; each one is imported explicitly because
        # walk_packages ignores import errors and a broken module must raise. Importing runs the
        # decorators, which label; the scan below is what registers.
        for info in pkgutil.walk_packages(package.__path__, prefix=f"{package.__name__}."):
            importlib.import_module(info.name)
        algorithms, declarations, requirements = _scan_plugin(plugin_name)

        # Each requirement is its own registration, completed before this package's snapshot, so
        # a failure here leaves already-registered requirements in place.
        for requirement in dict.fromkeys(requirements):
            try:
                register_plugin(requirement)
            except ModuleNotFoundError as exc:
                if exc.name == requirement:
                    raise ModuleNotFoundError(f"Could not import '{requirement}', required by '{plugin_name}'.", name=requirement) from exc
                raise

        algorithms_before = dict(Algorithm.instances)
        units_before = default_units_map()
        try:
            _load_plugin_variables(plugin_name)
            _register_scanned(algorithms, declarations)
            return [name for name in Algorithm.instances if name not in algorithms_before]
        except BaseException:
            # The modules this call imported stay cached, and that is what makes a retry work:
            # after the broken module is fixed, the retry's scan finds the same objects and
            # registers them.
            Algorithm.instances.clear()
            Algorithm.instances.update(algorithms_before)
            reset_default_units()
            extend_default_units_map(units_before)
            if plugin_name == _BUNDLED_PLUGIN:
                _BUNDLED_ALGORITHMS_DISCOVERED = False
            raise
    finally:
        _REGISTRATION_IN_PROGRESS.discard(plugin_name)


def _load_plugin_variables(plugin_name: str) -> None:
    """Read the plugin's own ``variables.yaml`` into the default units map, if it ships one."""
    variables_file = files(plugin_name).joinpath("variables.yaml")
    if variables_file.is_file():
        read_default_units_from_file(variables_file)


def _ensure_bundled_algorithms() -> None:
    """Register the bundled algorithms on the first use of the registry."""
    if not _BUNDLED_ALGORITHMS_DISCOVERED:
        register_plugin(_BUNDLED_PLUGIN)


def write_algorithms_yaml(filepath: Path) -> None:
    """Write a YAML listing of the registered algorithms, their inputs and their outputs.

    Args:
        filepath: the file to write.
    """
    _ensure_bundled_algorithms()
    data = {
        name: {"inputs": alg.required_input_keys, "optionals": alg.default_keys, "returns": alg.return_keys}
        for name, alg in Algorithm.instances.items()
    }
    with open(filepath, "w") as f:
        f.write("# Autogenerated by popcon_algorithms\n\n")
        f.write(yaml.dump(dict(sorted(data.items()))))


def discover_builtin_algorithms() -> list[str]:
    """Register every algorithm cfspopcon defines, by walking :mod:`cfspopcon.formulas`.

    The first use of the registry does this on its own; an explicit call is useful for surfacing
    any registration failure at a chosen point, e.g. the start of a batch job. Repeated calls
    change nothing.
    """
    return register_plugin(_BUNDLED_PLUGIN)


def algorithms_setting(variable: str) -> list[str]:
    """Names of the registered Algorithms whose outputs include ``variable``, sorted.

    Composites are excluded: each one contains a plain Algorithm which sets the variable anyway.
    """
    _ensure_bundled_algorithms()
    return sorted(name for name, alg in Algorithm.instances.items() if isinstance(alg, Algorithm) and variable in alg.return_keys)


def algorithms_using(variable: str) -> list[str]:
    """Names of the registered Algorithms whose inputs include ``variable``, sorted."""
    _ensure_bundled_algorithms()
    return sorted(name for name, alg in Algorithm.instances.items() if isinstance(alg, Algorithm) and variable in alg.input_keys)


class _AlgorithmRegistry:
    """The store of registered algorithms, keyed by name.

    ``registry["name"]`` returns the registered :class:`Algorithm` or :class:`CompositeAlgorithm`,
    ``registry.register(...)`` adds one, and ``"name" in registry`` or iteration lists the names.
    """

    def __getitem__(self, key: str) -> Algorithm | CompositeAlgorithm:
        """Look up a registered algorithm by name, registering the bundled algorithms first if needed.

        Args:
            key: the algorithm's registered name.

        Returns:
            The registered Algorithm or CompositeAlgorithm.

        Raises:
            TypeError: if the key is not a name.
            KeyError: if no algorithm of that name is registered, with a suggestion for near
                misses.
        """
        if not isinstance(key, str):
            raise TypeError("Index the algorithm registry with an algorithm name (str).")
        _ensure_bundled_algorithms()
        if key not in Algorithm.instances:
            raise KeyError(_algorithm_not_found_message(key))
        return Algorithm.instances[key]

    def register(self, algorithm: Algorithm | CompositeAlgorithm | GenericFunctionType, override: bool = False) -> None:
        """Register an algorithm, a composite, or a function labelled by :func:`algorithm`, under its name.

        Args:
            algorithm: the Algorithm or CompositeAlgorithm to register, or a labelled function.
            override: replace an already-registered algorithm of the same name.

        Raises:
            ValueError: if given anything but a named Algorithm, CompositeAlgorithm, or labelled
                function.
            RuntimeError: if the name is registered and ``override`` is not set.
        """
        algorithm = getattr(algorithm, "__popcon_algorithm__", algorithm)
        if not isinstance(algorithm, Algorithm | CompositeAlgorithm) or algorithm.name is None:
            raise ValueError("Only a named Algorithm or CompositeAlgorithm, or a labelled function, can be registered.")
        _register_algorithm(algorithm.name, algorithm, override)

    def __iter__(self) -> Iterator[str]:
        """Iterate over the registered algorithm names (also powers ``"name" in registry``)."""
        _ensure_bundled_algorithms()
        return iter(list(Algorithm.instances))


registry = _AlgorithmRegistry()
"""Registry accessor, where ``registry["name"]`` looks an algorithm up, ``registry.register`` adds one, and iteration lists the names."""
