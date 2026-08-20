# Changelog

All notable changes to cfspopcon will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

### Added

- **`discover_builtin_algorithms()`** — fill the registry by walking `cfspopcon.formulas` with `pkgutil`. A new `formulas/...` module needs no edit to `formulas/__init__.py`; a new subfolder still needs an (empty) `__init__.py`.
- **`discover_algorithms_in_package(package)`** — walk any package to register the algorithms beneath it, so a code built on cfspopcon can register its own.
- **`cfspopcon.registry` accessor** — `registry["name"]` returns an `Algorithm`, `registry[["a", "b"]]` builds a `CompositeAlgorithm`, and `"name" in registry` / iteration list the registered names.
- **`CompositeAlgorithm.register_from_list(keys, name)`** — declare a composite without looking its components up, for use in a module discovery imports in an arbitrary order. `build_pending_composites()` builds the declarations at the end of discovery, raising a `RuntimeError` which names the components nobody registered.
- **`override` flag** on `Algorithm(...)`, `Algorithm.from_single_function`, `@Algorithm.register_algorithm` and `CompositeAlgorithm(...)` — replace an already-registered algorithm of the same name instead of raising.
- **`.name` property** on `Algorithm` and `CompositeAlgorithm`, replacing `._name`.
- **`algorithms_setting(variable)` and `algorithms_using(variable)`** — which registered algorithms set, or take as an input, a given variable.
- **`calc_power_balance_from_input_P_aux` is now a registered algorithm**, so it can be listed in an `input.yaml`. It was previously reachable only as a module attribute.
- **JCH profile algorithms** — `calc_jch_profiles`, `calc_jch_pedestal_peaking`. (#139)
- **Profile-selection composite algorithms** — `calc_peaking_and_analytic_profiles`, `calc_peaking_and_prf_profiles`. (#139)
- **Radial-grid algorithm** — `define_radial_grid`, which provides `rho`. (#139)
- **Forward energy-confinement algorithms** — `calc_energy_confinement_time_from_scaling` and `calc_energy_confinement_time_from_stored_energy_and_input_power`, giving `energy_confinement_time` from a known input power. (#141)
- **`calc_H98y2`** — energy confinement time relative to the ITER98y2 scaling; adds the `H98y2` output. (#141)
- **Fixed-auxiliary-power balance** — `calc_input_power_for_fixed_auxiliary_power` and the `calc_power_balance_from_input_P_aux` composite. (#141)
- **`extend_default_units_map` is exported from `cfspopcon.unit_handling`** — the supported way for a downstream package to declare default units for variables of its own, since `read_default_units_from_file()` reads only cfspopcon's `variables.yaml` and takes no path argument. It was previously reachable only from `cfspopcon.unit_handling.default_units`.

### Changed

- **The algorithm registry is populated explicitly** — `import cfspopcon` registers nothing; call `cfspopcon.discover_builtin_algorithms()`. The `popcon` and `popcon_algorithms` commands do it for themselves.
- **The `formulas` subpackages no longer re-export their functions** — import from the module which defines one, e.g. `cfspopcon.formulas.geometry.analytical.calc_plasma_volume`. Submodules of `cfspopcon.formulas` are imported on first attribute access rather than eagerly.
- **Composite algorithms are no longer module-level variables** — `calc_peaking_and_analytic_profiles`, `calc_peaking_and_prf_profiles` and `calc_power_balance_from_input_P_aux` are declared, and reached through the registry (`registry["name"]`).
- **`set_up_impurity_concentration_array` moved** to `cfspopcon.formulas.impurities.impurity_array_helpers`.
- **Discovery fails loudly** — a module which will not import, or a composite naming an algorithm nobody registers, raises. A failed walk cannot be retried in the same process, since whatever it registered is still registered; start a new one.
- **`skip_registration=True` no longer raises on a duplicate name**, so a variant of a registered algorithm can be built without touching the registry.
- **Missing and unused inputs come with a hint** — which registered algorithms set a missing variable, and the nearest match for an unused one.
- **The unknown-algorithm error message** describes discovery, and suggests the nearest registered name, instead of telling you to add an import to `cfspopcon/__init__.py`. The duplicate-registration message names the `override` / `skip_registration` escape hatches.
- **Profile form is selected by algorithm** — list a `calc_peaking_and_*_profiles` composite instead of setting the `density_profile_form` / `temp_profile_form` inputs. (#139)
- **`calc_analytic_profiles`, `calc_prf_profiles` algorithms** now take `rho` as an input and no longer return it; the `npoints` argument is removed. (#139)
- **`wraps_ufunc`** infers `output_core_dims` from the number of return units, so multi-return functions no longer need to pass it explicitly. (#141)

### Fixed

- **A named `CompositeAlgorithm`'s docstring** listed only its name. Operator precedence between the conditional expression and the implicitly concatenated strings meant the components, inputs and outputs were built only for an unnamed composite; since every registered composite is named, they were missing everywhere they were used.

### Removed

- **`cfspopcon.AtomicData`** — import it from `cfspopcon.formulas.atomic_data` instead. Importing it at the top level registered an algorithm merely by importing cfspopcon.
- **`CompositeAlgorithm.from_list`** — index the registry for an unnamed composite (`registry[["a", "b"]]`), or use `CompositeAlgorithm.register_from_list` to register a named one.
- **`calc_peaked_profiles`, `calc_1D_plasma_profiles` algorithms** — replaced by `calc_peaking_and_analytic_profiles` / `calc_peaking_and_prf_profiles`. (#139)
- **`density_profile_form`, `temp_profile_form` inputs** — and with them, mixed density/temperature profile forms. (#139)
