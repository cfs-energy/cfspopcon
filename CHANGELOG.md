# Changelog

All notable changes to cfspopcon will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

This release makes cfspopcon extensible: plugin packages can add algorithms, composites, and
variables, and a case's input file can use them like the built-in ones. To support this,
importing cfspopcon runs no registration code at all. This is a breaking release, planned as a
new major version: breaking entries are marked **breaking**, and every entry under Removed is
breaking.

### Added

- **Plugins**: `register_plugin("my_popcon_plugin")` registers a package built on cfspopcon: the default units in its `variables.yaml`, the algorithms defined in its modules, and the composites they declare. The plugin's `__init__.py` may be empty, and the bundled algorithms are always registered before any plugin. See the new "Authoring a Plugin" documentation page.
- **`plugins` section in `input.yaml`**: a case lists the plugins it uses, registered in order before the `algorithms` names are resolved. `plugins` becomes a reserved top-level input-file key.
- **`popcon_algorithms --plugin`** (repeatable): list a plugin's algorithms alongside the built-in ones.
- **`__popcon_requires__`**: a plugin names the plugins whose algorithms its composites build on, as a module-level tuple; each requirement is registered first, and a circular requirement raises.
- **`cfspopcon.registry`**: `registry["name"]` returns the registered `Algorithm`, `registry.register(...)` adds an algorithm, a composite, or a labelled function, and `"name" in registry` or iteration lists the registered names.
- **`CompositeAlgorithm.declare(keys, name)`**: declare a composite by the names of its components before those exist; it is built and registered with its plugin, and a missing component is a `RuntimeError` naming it. `override=True` replaces a registered algorithm of the composite's name.
- **`override` flag** on `@Algorithm.register_algorithm`, `Algorithm(...)`, `Algorithm.from_single_function` and `registry.register`: deliberately replace a registered algorithm of the same name.
- **`algorithms_setting(variable)` and `algorithms_using(variable)`**: which registered algorithms set, or take as an input, a given variable.
- **`.name` property** on `Algorithm` and `CompositeAlgorithm`, replacing the private `._name`.
- **`extend_default_units_map` exported from `cfspopcon.unit_handling`**: declare default units for new variables from code; `read_default_units_from_file` accepts a path, so a plugin can also ship its own `variables.yaml`.
- **JCH profile algorithms**: `calc_jch_profiles`, `calc_jch_pedestal_peaking`. (#139)
- **Profile-selection composite algorithms**: `calc_peaking_and_analytic_profiles`, `calc_peaking_and_prf_profiles`. (#139)
- **Radial-grid algorithm**: `define_radial_grid`, which provides `rho`. (#139)
- **Forward energy-confinement algorithms**: `calc_energy_confinement_time_from_scaling` and `calc_energy_confinement_time_from_stored_energy_and_input_power`, giving `energy_confinement_time` from a known input power. (#141)
- **`calc_H98y2`**: energy confinement time relative to the ITER98y2 scaling; adds the `H98y2` output. (#141)
- **Fixed-auxiliary-power balance**: `calc_input_power_for_fixed_auxiliary_power` and the `calc_power_balance_from_input_P_aux` composite, registered so a case can list it. (#141)

### Changed

- **Importing cfspopcon registers nothing** (**breaking**): the registry fills on its first use, or at a chosen point with `discover_builtin_algorithms()`. Formula modules import without side effects, so cfspopcon works as a plain library of functions, and a new `formulas` module is picked up with no edit to any `__init__.py` (a new subfolder still needs one, even empty).
- **Construction and decoration register nothing** (**breaking**): an `Algorithm`, a decorated function, or a `CompositeAlgorithm` enters the registry only through `register_plugin` or `registry.register`.
- **The bundled composites are declarations** (**breaking**): `calc_peaking_and_analytic_profiles`, `calc_peaking_and_prf_profiles` and `calc_power_balance_from_input_P_aux` are declarations at module level; the runnable composite is reached through the registry, e.g. `registry["calc_power_balance_from_input_P_aux"]`.
- **Registration fails loudly and rolls back**: a module which does not import, or a composite naming an unregistered algorithm, raises; a failed `register_plugin` restores the registries, so the plugin can be fixed and registered again in the same session. Units defined with `ureg.define` are the one thing which cannot be rolled back.
- **A composite resolves at the end of its plugin's registration**: it may name any algorithm registered by then; register a plugin after the plugins it builds on, or declare them with `__popcon_requires__`.
- **A variable's default units cannot change once defined** (**breaking**): re-declaring identical units is allowed; a change raises a `ValueError` naming the variable.
- **Errors and warnings suggest the fix**: a missing input names the registered algorithms which set it, an unused input suggests the nearest matching name, and an unknown algorithm suggests the nearest registered name or registering the plugin which provides it.
- **Profile form is selected by algorithm** (**breaking**): list a `calc_peaking_and_*_profiles` composite instead of setting the `density_profile_form` / `temp_profile_form` inputs. (#139)
- **`calc_analytic_profiles`, `calc_prf_profiles`** (**breaking**): take `rho` as an input instead of returning it; the `npoints` argument is removed. (#139)
- **`wraps_ufunc`** infers `output_core_dims` from the number of return units, so multi-return functions need not pass it explicitly. (#141)

### Fixed

- **A named `CompositeAlgorithm`'s docstring** listed only its name; it now lists the components, inputs and outputs.

### Removed

- **`skip_registration`** from `Algorithm(...)` and `Algorithm.from_single_function`: construction registers nothing, so the flag is meaningless. Delete the argument at call sites.
- **`register` / `override` arguments of `CompositeAlgorithm(...)`, and the `register` argument of `CompositeAlgorithm.from_list`**: constructors register nothing; register a composite with `registry.register(composite)`, passing `override=True` to replace.
- **`Algorithm.get_algorithm`, `Algorithm.algorithms`, `Algorithm.write_yaml`**: the registry owns its verbs: look up with `registry["name"]`, list with `list(registry)`, and write the YAML listing with `popcon_algorithms`.
- **`set_by` / `used_by` fields in `variables.yaml`**: write-only bookkeeping, and the bulk of the file; `algorithms_setting(variable)` / `algorithms_using(variable)` answer from the live registry instead.
- **`calc_peaked_profiles`, `calc_1D_plasma_profiles` algorithms**: replaced by `calc_peaking_and_analytic_profiles` / `calc_peaking_and_prf_profiles`. (#139)
- **`density_profile_form`, `temp_profile_form` inputs**: and with them, mixed density/temperature profile forms. (#139)
