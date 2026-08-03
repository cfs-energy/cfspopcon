"""Defines a class for different POPCON algorithms, and the discovery which registers them.

To use cfspopcon's algorithms, all that is needed is::

    import cfspopcon

    cfspopcon.discover_builtin_algorithms()

That walks :mod:`cfspopcon.formulas` with ``pkgutil``, registering every algorithm cfspopcon defines
and building every composite, after which they can be looked up by name in ``cfspopcon.registry``.
Importing cfspopcon on its own registers nothing. A package built on cfspopcon adds algorithms of its
own the same way, by calling :func:`discover_algorithms_in_package` on itself.

The walk registers algorithms, but only *declares* composites (see
:meth:`CompositeAlgorithm.register_from_list`), so the order it happens to visit modules in does not
matter: :func:`build_pending_composites` builds the declarations once the walk has finished.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from difflib import get_close_matches
from functools import wraps
from pathlib import Path  # noqa: TC003
from types import ModuleType  # noqa: TC003
from typing import Any, ClassVar
from warnings import warn

import xarray as xr
import yaml

from .unit_handling import Quantity, convert_to_default_units, ureg

LabelledReturnFunctionType = Callable[..., dict[str, Any]]
GenericFunctionType = Callable[..., Any]

#: Composites declared by :meth:`CompositeAlgorithm.register_from_list` but not yet built, as
#: (name, component names). Drained by :func:`build_pending_composites`.
_pending_composites: list[tuple[str, list[str]]] = []

#: How many walks are in progress, so a nested one leaves the composite build to the outermost.
_walk_depth = 0


def _not_found_message(key: str) -> str:
    """Explain as specifically as possible why an algorithm name did not resolve."""
    if any(name == key for name, _ in _pending_composites):
        return (
            f"algorithm '{key}' is declared but not built yet: composites are built only once discovery has "
            "finished, so one cannot be looked up from a module that discovery is importing."
        )

    if not Algorithm.instances:
        return (
            f"algorithm '{key}' not found: the registry is empty because discovery has not run. "
            "Call cfspopcon.discover_builtin_algorithms() first."
        )

    close_matches = get_close_matches(key, Algorithm.algorithms(), n=1)
    if close_matches:
        return f"algorithm '{key}' not found. Did you mean '{close_matches[0]}'?"

    return (
        f"algorithm '{key}' not found. discover_builtin_algorithms registers those under cfspopcon.formulas; "
        "one in another package needs a discover_algorithms_in_package call for it. "
        "Run popcon_algorithms to list what is registered."
    )


def _register(name: str, algorithm: Algorithm | CompositeAlgorithm, override: bool) -> None:
    """Add an algorithm to the registry, refusing to silently replace one of the same name."""
    if name in Algorithm.instances and not override:
        raise RuntimeError(
            f"Algorithm '{name}' is already registered. Pass override=True to replace it, or build it "
            "without registering (skip_registration=True for an Algorithm, register=False for a composite)."
        )
    Algorithm.instances[name] = algorithm


class Algorithm:
    """A class which handles the input and output of POPCON algorithms."""

    #: The registered algorithms, keyed by name. Empty until :func:`discover_builtin_algorithms` has run.
    instances: ClassVar[dict[str, Algorithm | CompositeAlgorithm]] = dict()

    def __init__(
        self,
        function: LabelledReturnFunctionType,
        return_keys: list[str],
        name: str | None = None,
        skip_registration: bool = False,
        override: bool = False,
    ):
        """Initialise an Algorithm.

        Args:
            function: a callable function
            return_keys: the arguments which are returned from the function
            name: Descriptive name for algorithm
            skip_registration: construct the Algorithm without adding it to 'instances' (useful for
                testing, or for a coexisting variant of an already-registered algorithm)
            override: replace an already-registered algorithm of this name, rather than raising
        """
        self._function = function
        self._name = self._function.__name__ if name is None else name

        if not skip_registration:
            _register(self._name, self, override)

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
        """The name this algorithm is registered under (the function's name unless overridden)."""
        return self._name

    def __repr__(self) -> str:
        """Return a simple string description of the Algorithm."""
        return f"Algorithm: {self._name}"

    def __call__(self, *args: Any, return_labelled_dictionary: bool = False, **kwargs: Any) -> Any:
        """Call the algorithm like a function, returning its outputs directly.

        By default, returns the value(s) in ``return_keys`` order: one value, or
        a tuple for several outputs.
        With ``return_labelled_dictionary=True``, returns the ``{return_key: value}`` mapping.

        Example::

            algorithm = Algorithm.get_algorithm("algorithm_name")
            outputs = algorithm(input_a=..., input_b=...)
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
        skip_registration: bool = False,
        override: bool = False,
    ) -> Algorithm:
        """Build an Algorithm which wraps a single function."""
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
            skip_registration=skip_registration,
            override=override,
        )

    @classmethod
    def register_algorithm(
        cls, return_keys: list[str], name: str | None = None, skip_unit_conversion: bool = False, override: bool = False
    ) -> GenericFunctionType:
        """Decorate a function and turn it into an Algorithm. Usage: @Algorithm.register_algorithm(return_keys=["..."])."""

        def function_wrapper(func: GenericFunctionType) -> GenericFunctionType:
            Algorithm.from_single_function(
                func,
                return_keys=return_keys,
                name=name if name is not None else func.__name__,
                skip_unit_conversion=skip_unit_conversion,
                override=override,
            )
            return func

        return function_wrapper

    @classmethod
    def empty(cls) -> Algorithm:
        """Makes a 'do nothing' algorithm, in case you don't want to use the algorithm functionality."""

        def do_nothing() -> dict[str, Any]:
            result_dict: dict[str, Any] = {}
            return result_dict

        return cls(do_nothing, return_keys=[], name="empty", skip_registration=True)

    def validate_inputs(self, configuration: dict | xr.Dataset, quiet: bool = False, raise_error_on_missing_inputs: bool = False) -> bool:
        """Check that all required inputs are defined, and warn if inputs are unused."""
        return _validate_inputs(self, configuration, quiet=quiet, raise_error_on_missing_inputs=raise_error_on_missing_inputs)

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
    def get_algorithm(cls, key: str) -> Algorithm | CompositeAlgorithm:
        """Retrieves an algorithm by name."""
        if key not in cls.instances:
            raise KeyError(_not_found_message(key))

        return cls.instances[key]


class CompositeAlgorithm:
    """A class which combined multiple Algorithms into a single object which behaves like an Algorithm."""

    def __init__(
        self,
        algorithms: Sequence[Algorithm | CompositeAlgorithm],
        name: str | None = None,
        register: bool = False,
        override: bool = False,
    ):
        """Initialise a CompositeAlgorithm, combining several other Algorithms.

        Args:
            algorithms: a list of Algorithms, in the order that they should be executed.
            name: a name used to refer to the composite algorithm.
            register: flag register a named CompositeAlgorithm to 'Algorithm.instances' (ignored if name = None)
            override: replace an already-registered algorithm of this name, rather than raising
        """
        if not (isinstance(algorithms, Sequence) and all(isinstance(alg, Algorithm | CompositeAlgorithm) for alg in algorithms)):
            raise TypeError("Should pass a list of algorithms or composites to CompositeAlgorithm.")

        self.algorithms: list[Algorithm] = []

        if (name is not None) and (register):
            _register(name, self, override)

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
    def register_from_list(cls, keys: list[str], name: str) -> None:
        """Declare a named CompositeAlgorithm, to be built once its components are registered.

        Nothing is looked up here, so a module can declare a composite whichever order discovery
        happens to import it in. :func:`build_pending_composites` builds the declarations, and is
        called for you at the end of discovery. To build one immediately from already-registered
        algorithms, index the registry instead: ``registry[["a", "b"]]``.
        """
        _pending_composites.append((name, list(keys)))

    def _make_docstring(self) -> str:
        """Makes a doc-string detailing the function inputs and outputs."""
        components = f"[{', '.join(alg.name for alg in self.algorithms)}]"

        return_string = (
            f"CompositeAlgorithm: {self._name}\n"
            if self._name is not None
            else "CompositeAlgorithm\n"
            f"Composed of {components}\n"
            f"Inputs:\n{', '.join(self.input_keys)}\n"
            f"Outputs:\n{', '.join(self.return_keys)}"
        )
        return return_string

    @property
    def name(self) -> str | None:
        """The name this composite is registered under, or None for an unnamed composite."""
        return self._name

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
        """Check that all required inputs are defined, and warn if inputs are unused."""
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


def build_pending_composites() -> None:
    """Build every composite declared by :meth:`CompositeAlgorithm.register_from_list`.

    A composite may be built from other composites, so each pass builds whichever declarations have
    all of their components registered by now, and repeats. A pass which can build nothing raises,
    naming the missing components, and leaves the declarations pending so that a later walk which
    registers them can still build them. Every pass therefore either builds something or raises,
    which is what terminates the loop.
    """
    while _pending_composites:
        ready = [(name, keys) for name, keys in _pending_composites if all(key in Algorithm.instances for key in keys)]
        if not ready:
            unresolved = "; ".join(
                f"'{name}' is missing [{', '.join(k for k in keys if k not in Algorithm.instances)}]" for name, keys in _pending_composites
            )
            raise RuntimeError(f"Could not build the composite algorithms: {unresolved}.")

        for name, keys in ready:
            CompositeAlgorithm([Algorithm.instances[key] for key in keys], name=name, register=True)
            _pending_composites.remove((name, keys))  # drop it only once it has actually been built


@contextmanager
def deferred_composite_build() -> Iterator[None]:
    """Treat everything registered inside the block as one unit, building composites when it exits.

    Blocks nest: only the outermost one builds, so a composite declared in one package may name an
    algorithm registered by another, provided both are registered inside the same outermost block.
    This is what :func:`discover_algorithms_in_packages` and plugin registration use to walk several
    packages without each walk building on its own way out.
    """
    global _walk_depth  # noqa: PLW0603
    _walk_depth += 1
    try:
        yield
    finally:
        _walk_depth -= 1
    if not _walk_depth:
        build_pending_composites()


def discover_algorithms_in_packages(*packages: ModuleType | str) -> None:
    """Register every algorithm defined anywhere beneath each of ``packages``, as a single unit.

    Lets a package which builds on cfspopcon register all of its algorithms without importing each
    module by hand. A new subfolder needs an ``__init__.py``: a directory without one is not walked.

    All of the walks share one :func:`deferred_composite_build` block, so a composite may span the
    packages given, in either direction. The composites are built once every package has been walked,
    and may name anything registered by then -- including cfspopcon's own algorithms, which means
    :func:`discover_builtin_algorithms` has to have run first.

    A package which fails to import aborts the remaining ones, and nothing it registered is undone.
    Registering plugins is the caller that cares; see :func:`cfspopcon.plugins.register_plugins`, which rolls
    the whole set back.
    """
    with deferred_composite_build():
        for package in packages:
            # Inside the block, since importing the package may itself walk.
            module = importlib.import_module(package) if isinstance(package, str) else package
            for info in pkgutil.walk_packages(module.__path__, prefix=f"{module.__name__}."):
                importlib.import_module(info.name)


def discover_algorithms_in_package(package: ModuleType | str) -> None:
    """Register every algorithm defined anywhere beneath ``package``, given as a module or its name.

    Single-package spelling of :func:`discover_algorithms_in_packages`.
    """
    discover_algorithms_in_packages(package)


def discover_builtin_algorithms() -> None:
    """Register every algorithm cfspopcon defines, by walking :mod:`cfspopcon.formulas`.

    Call this before using the registry: ``import cfspopcon`` deliberately registers nothing.
    Repeated calls are cheap and change nothing, since the modules walked are already imported.
    """
    from . import formulas

    discover_algorithms_in_packages(formulas)


def registered_algorithms() -> dict[str, Algorithm | CompositeAlgorithm]:
    """Return a copy of the registry, mapping name to algorithm.

    Keyed by name *and* holding the objects, so a caller comparing two snapshots can tell a name
    which was added from one which was replaced -- see the ``override`` flag on :class:`Algorithm`.
    """
    return dict(Algorithm.instances)


def pending_composites() -> list[tuple[str, list[str]]]:
    """Return a copy of the composites declared but not yet built, as (name, component names)."""
    return [(name, list(keys)) for name, keys in _pending_composites]


def restore_registry(algorithms: dict[str, Algorithm | CompositeAlgorithm], pending: list[tuple[str, list[str]]]) -> None:
    """Replace the registry and the pending declarations with the given snapshots.

    Both are needed together: undoing registrations without undoing declarations leaves a composite
    pending whose components have gone, and the next build would raise naming it.
    """
    Algorithm.instances.clear()
    Algorithm.instances.update(algorithms)
    _pending_composites[:] = pending


class _AlgorithmRegistry:
    """Provides indexed access to the algorithm registry.

    ``registry["name"]`` returns the named :class:`Algorithm`; ``registry[["a", "b"]]`` returns an
    unregistered :class:`CompositeAlgorithm` which executes those algorithms in the order given.
    """

    def __getitem__(self, key: str | list[str] | tuple[str, ...]) -> Algorithm | CompositeAlgorithm:
        """Look up an Algorithm by name, or build a CompositeAlgorithm from a list/tuple of names."""
        if isinstance(key, str):
            return Algorithm.get_algorithm(key)
        if isinstance(key, (list, tuple)):
            return CompositeAlgorithm([Algorithm.get_algorithm(name) for name in key])
        raise TypeError("Index the algorithm registry with a name (str) or a list/tuple of names.")

    def __iter__(self) -> Iterator[str]:
        """Iterate over the registered algorithm names (also powers ``"name" in registry``)."""
        return iter(Algorithm.algorithms())


registry = _AlgorithmRegistry()
"""Registry accessor, where ``registry["name"]`` gives an Algorithm and ``registry[["a", "b"]]`` a CompositeAlgorithm."""
