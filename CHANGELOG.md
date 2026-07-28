# Changelog

All notable changes to cfspopcon will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

This will be released as **v9.0.0**; the version in `pyproject.toml` has been bumped ahead of the
release because of the breaking changes listed below. (#147)

### Added

- **Automatic algorithm discovery** — `cfspopcon.formulas` is walked with `pkgutil` on first use, so adding a `formulas/...` module registers its algorithms with no edit to `formulas/__init__.py`. A new subfolder still needs an (empty) `__init__.py`; a directory without one is not walked. (#147)
- **`discover_algorithms_in_package(package)`** — walk any package to register the algorithms defined beneath it, for codes which build on cfspopcon. (#147)
- **`discover_builtin_algorithms()`** — populate the registry now rather than on the first query, for callers that read `Algorithm.instances` directly instead of going through the registry accessors. It runs the same discovery a query would, so entry-point providers load too. (#147)
- **`cfspopcon.algorithms` entry-point group** — an installed distribution can contribute algorithms with no cfspopcon-side import, and this is the only route the command-line interface sees. The target may be a module (imported for its registration side effects) or a callable taking no arguments. It is imported, not walked, so point it at the module which registers, or at a callable which walks your package. (#147)
- **`cfspopcon.registry` accessor** — `registry["name"]` returns an `Algorithm`, `registry[["a", "b"]]` builds a `CompositeAlgorithm`, and `"name" in registry` / iteration list the registered names. Named `registry` rather than `algorithms` to keep it distinct from `Algorithm.algorithms()` and from a composite's own `.algorithms`. (#147)
- **`override` flag** on `Algorithm(...)`, `Algorithm.from_single_function`, `@Algorithm.register_algorithm` and `CompositeAlgorithm(...)` — deliberately replace an already-registered algorithm of the same name instead of raising. (#147)
- **`cfspopcon.algorithm_class.build_pending_composites()`** — build the declared-but-not-yet-built composites. Called for you at the end of discovery; exposed because it is part of the discovery contract. (#147)
- **`CompositeAlgorithm.register_from_list(keys, name)`** — declare a composite without looking its components up, so a module can declare one whichever order discovery imports it in. The declarations are built after the walk, repeating until composites built from other composites are satisfied; anything that can never be built raises a `RuntimeError` naming the missing components. (#147)
- **`calc_power_balance_from_input_P_aux` is now a registered algorithm**, so it can be listed in an `input.yaml` and appears in `algorithms.yaml`. It was previously reachable only as a module attribute. (#147)
- **JCH profile algorithms** — `calc_jch_profiles`, `calc_jch_pedestal_peaking`. (#139)
- **Profile-selection composite algorithms** — `calc_peaking_and_analytic_profiles`, `calc_peaking_and_prf_profiles`. (#139)
- **Radial-grid algorithm** — `define_radial_grid`, which provides `rho`. (#139)
- **Forward energy-confinement algorithms** — `calc_energy_confinement_time_from_scaling` and `calc_energy_confinement_time_from_stored_energy_and_input_power`, giving `energy_confinement_time` from a known input power. (#141)
- **`calc_H98y2`** — energy confinement time relative to the ITER98y2 scaling; adds the `H98y2` output. (#141)
- **Fixed-auxiliary-power balance** — `calc_input_power_for_fixed_auxiliary_power` and the `calc_power_balance_from_input_P_aux` composite. (#141)

### Changed

- **The algorithm registry is populated lazily** — discovery runs once, on the first query through `Algorithm.algorithms()`, `Algorithm.get_algorithm()`, `Algorithm.write_yaml()` or the `algorithms` accessor. Code which reads `Algorithm.instances` directly must call `discover_builtin_algorithms()` first. (#147)
- **Composite algorithms are no longer module-level variables** — `calc_peaking_and_analytic_profiles`, `calc_peaking_and_prf_profiles` and `calc_power_balance_from_input_P_aux` are declared rather than built at import time, so they are reached through the registry (`registry["name"]`) instead of imported from their defining module. (#147)
- **`cfspopcon.formulas` submodules are imported on first attribute access** rather than eagerly, so `cfspopcon.formulas.geometry` keeps working. Its `__all__` is now discovered by scanning the package instead of being hand-maintained, so `from cfspopcon.formulas import *` still exports exactly the submodules. (#147)
- **`skip_registration=True` no longer raises on a duplicate name** — it is honoured before the duplicate-name check, so a variant of an already-registered algorithm can be constructed without touching the registry. (#147)
- **The unknown-algorithm error message** now describes discovery instead of telling you to add an import to `cfspopcon/__init__.py`. The duplicate-registration message now names the `override` / `skip_registration` escape hatches, and `Algorithm` and `CompositeAlgorithm` report a collision the same way. (#147)
- **Discovery is all-or-nothing.** A module which will not import, a broken `cfspopcon.algorithms` entry point, or a composite naming an algorithm nobody registers fails the query that triggered discovery, and every later query re-raises that same error rather than handing back a half-filled registry. (#147)
- **`discover_algorithms_in_package()` runs full discovery first** (cfspopcon's own algorithms and any entry-point providers), so a composite declared by the walked package can be built from them, and a registry query made by a module during the walk no longer builds composites early. (#147)
- **Profile form is selected by algorithm** — list a `calc_peaking_and_*_profiles` composite instead of setting the `density_profile_form` / `temp_profile_form` inputs. (#139)
- **`calc_analytic_profiles`, `calc_prf_profiles` algorithms** now take `rho` as an input and no longer return it; the `npoints` argument is removed. (#139)
- **`wraps_ufunc`** infers `output_core_dims` from the number of return units, so multi-return functions no longer need to pass it explicitly. (#141)

### Removed

- **`CompositeAlgorithm.from_list`** — to build an unnamed composite from a list of names, index the registry instead: `algorithms[["a", "b"]]`. To register a named one, use `CompositeAlgorithm.register_from_list`; for a named but unregistered composite, construct `CompositeAlgorithm([...], name=...)` directly. (#147)
- **`calc_peaked_profiles`, `calc_1D_plasma_profiles` algorithms** — replaced by `calc_peaking_and_analytic_profiles` / `calc_peaking_and_prf_profiles`. (#139)
- **`density_profile_form`, `temp_profile_form` inputs** — and with them, mixed density/temperature profile forms. (#139)
