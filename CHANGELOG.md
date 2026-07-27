# Changelog

All notable changes to cfspopcon will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

### Added

- **`discover_builtin_algorithms()`** — populate the registry by walking `cfspopcon.formulas` with `pkgutil`. Adding a `formulas/...` module therefore registers its algorithms with no edit to `formulas/__init__.py`, though a new subfolder still needs an (empty) `__init__.py`: a directory without one is not walked.
- **`discover_algorithms_in_package(package)`** — walk any package to register the algorithms beneath it. This is how a code built on cfspopcon adds algorithms of its own: call `discover_builtin_algorithms()` and then this, from that code's own command line, before reading an input file.
- **`cfspopcon.registry` accessor** — `registry["name"]` returns an `Algorithm`, `registry[["a", "b"]]` builds a `CompositeAlgorithm`, and `"name" in registry` / iteration list the registered names. Named `registry` rather than `algorithms` to keep it distinct from `Algorithm.algorithms()` and from a composite's own `.algorithms`.
- **`CompositeAlgorithm.register_from_list(keys, name)`** — declare a composite without looking its components up, so a module can declare one whichever order discovery imports it in. `cfspopcon.algorithm_class.build_pending_composites()` builds the declarations, and is called at the end of discovery; a declaration that can never be built raises a `RuntimeError` naming the missing components.
- **`override` flag** on `Algorithm(...)`, `Algorithm.from_single_function`, `@Algorithm.register_algorithm` and `CompositeAlgorithm(...)` — deliberately replace an already-registered algorithm of the same name instead of raising.
- **`calc_power_balance_from_input_P_aux` is now a registered algorithm**, so it can be listed in an `input.yaml` and appears in `popcon_algorithms.yaml`. It was previously reachable only as a module attribute.
- **JCH profile algorithms** — `calc_jch_profiles`, `calc_jch_pedestal_peaking`. (#139)
- **Profile-selection composite algorithms** — `calc_peaking_and_analytic_profiles`, `calc_peaking_and_prf_profiles`. (#139)
- **Radial-grid algorithm** — `define_radial_grid`, which provides `rho`. (#139)
- **Forward energy-confinement algorithms** — `calc_energy_confinement_time_from_scaling` and `calc_energy_confinement_time_from_stored_energy_and_input_power`, giving `energy_confinement_time` from a known input power. (#141)
- **`calc_H98y2`** — energy confinement time relative to the ITER98y2 scaling; adds the `H98y2` output. (#141)
- **Fixed-auxiliary-power balance** — `calc_input_power_for_fixed_auxiliary_power` and the `calc_power_balance_from_input_P_aux` composite. (#141)
- **`register_plugin()`** — register an external plugin package, which registers its own algorithms, variables and units, with clash detection, rollback and a report of what was added. (#148)
- **`extend_default_units_map` is exported from `cfspopcon.unit_handling`** — the supported way for a downstream package to declare default units for variables of its own, since `read_default_units_from_file()` reads only cfspopcon's `variables.yaml` and takes no path argument. It was previously reachable only from `cfspopcon.unit_handling.default_units`.

### Changed

- **The algorithm registry is populated explicitly** — `import cfspopcon` registers nothing; call `cfspopcon.discover_builtin_algorithms()`. Until then the registry is empty and a lookup says so, rather than reporting an algorithm as missing from a registry that was never filled. The `popcon` and `popcon_algorithms` commands discover for themselves.
- **Composite algorithms are no longer module-level variables** — `calc_peaking_and_analytic_profiles`, `calc_peaking_and_prf_profiles` and `calc_power_balance_from_input_P_aux` are declared rather than built at import time, so they are reached through the registry (`registry["name"]`) instead of imported from their defining module.
- **`cfspopcon.formulas` submodules are imported on first attribute access** rather than eagerly, so `cfspopcon.formulas.geometry` keeps working. Its `__all__` is now discovered by scanning the package instead of being hand-maintained, so `from cfspopcon.formulas import *` still exports exactly the submodules.
- **Discovery fails loudly.** A module which will not import, or a composite naming an algorithm nobody registers, raises. Retrying a failed walk in the same process is not supported: whatever the abandoned attempt registered is still registered, so the retry reports a name collision rather than the original cause — fix the cause and start a new process.
- **`discover_algorithms_in_package()` walks only the package it is given,** building the composites it declares when the walk finishes — so one naming a cfspopcon algorithm needs `discover_builtin_algorithms()` to have run first. A walk nested inside another leaves that build to the outermost one, so two packages can declare composites spanning both.
- **`skip_registration=True` no longer raises on a duplicate name** — it is honoured before the duplicate-name check, so a variant of an already-registered algorithm can be constructed without touching the registry.
- **The unknown-algorithm error message** now describes discovery instead of telling you to add an import to `cfspopcon/__init__.py`. The duplicate-registration message now names the `override` / `skip_registration` escape hatches, and `Algorithm` and `CompositeAlgorithm` report a collision the same way.
- **Profile form is selected by algorithm** — list a `calc_peaking_and_*_profiles` composite instead of setting the `density_profile_form` / `temp_profile_form` inputs. (#139)
- **`calc_analytic_profiles`, `calc_prf_profiles` algorithms** now take `rho` as an input and no longer return it; the `npoints` argument is removed. (#139)
- **`wraps_ufunc`** infers `output_core_dims` from the number of return units, so multi-return functions no longer need to pass it explicitly. (#141)

### Removed

- **`cfspopcon.AtomicData`** — import it from `cfspopcon.formulas.atomic_data` instead. Importing it at the top level was the one thing that registered an algorithm merely by importing cfspopcon, which made an un-discovered registry look like a populated one holding a single algorithm.
- **`CompositeAlgorithm.from_list`** — to build an unnamed composite from a list of names, index the registry instead: `registry[["a", "b"]]`. To register a named one, use `CompositeAlgorithm.register_from_list`; for a named but unregistered composite, construct `CompositeAlgorithm([...], name=...)` directly.
- **`calc_peaked_profiles`, `calc_1D_plasma_profiles` algorithms** — replaced by `calc_peaking_and_analytic_profiles` / `calc_peaking_and_prf_profiles`. (#139)
- **`density_profile_form`, `temp_profile_form` inputs** — and with them, mixed density/temperature profile forms. (#139)
