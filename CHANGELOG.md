# Changelog

All notable changes to cfspopcon will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

This will be released as **v9.0.0**; the version in `pyproject.toml` has been bumped ahead of the
release because of the breaking changes listed below. (#147)

### Added

- **Automatic algorithm discovery** — `cfspopcon.formulas` is walked with `pkgutil` on first use, so adding a `formulas/...` module registers its algorithms with no `__init__.py` edit. (#147)
- **`discover_algorithms_in_package(package)`** — walk any package to register the algorithms defined beneath it, for codes which build on cfspopcon. (#147)
- **`discover_builtin_algorithms()`** — register cfspopcon's own algorithms explicitly, for callers that read `Algorithm.instances` directly instead of going through the registry accessors. (#147)
- **`cfspopcon.algorithms` entry-point group** — an installed distribution can contribute algorithms with no cfspopcon-side import. The entry-point target may be a module (imported for its registration side effects) or a callable (invoked to register). (#147)
- **`algorithms` registry accessor** — `algorithms["name"]` returns an `Algorithm`, `algorithms[["a", "b"]]` builds a `CompositeAlgorithm`, and `"name" in algorithms` / iteration list the registered names. (#147)
- **`override` flag** on `Algorithm(...)`, `Algorithm.from_single_function`, `@Algorithm.register_algorithm` and `CompositeAlgorithm(...)` — deliberately replace an already-registered algorithm of the same name instead of raising. (#147)
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
- **Composite algorithms are no longer module-level variables** — `calc_peaking_and_analytic_profiles`, `calc_peaking_and_prf_profiles` and `calc_power_balance_from_input_P_aux` are declared rather than built at import time, so they are reached through the registry (`algorithms["name"]`) instead of imported from their defining module. (#147)
- **`cfspopcon.formulas` submodules are imported on first attribute access** rather than eagerly, so `cfspopcon.formulas.geometry` keeps working without a hand-maintained import list. (#147)
- **`skip_registration=True` no longer raises on a duplicate name** — it is honoured before the duplicate-name check, so a variant of an already-registered algorithm can be constructed without touching the registry. (#147)
- **The duplicate-registration and unknown-algorithm error messages** now describe discovery instead of telling you to add an import to `cfspopcon/__init__.py`. `Algorithm` and `CompositeAlgorithm` report a name collision the same way. (#147)
- **A failed discovery can be retried.** If a module raises partway through the walk, whatever it registered before raising is rolled back, so the retry re-runs that module cleanly rather than colliding with its own leftovers. (#147)
- **Profile form is selected by algorithm** — list a `calc_peaking_and_*_profiles` composite instead of setting the `density_profile_form` / `temp_profile_form` inputs. (#139)
- **`calc_analytic_profiles`, `calc_prf_profiles` algorithms** now take `rho` as an input and no longer return it; the `npoints` argument is removed. (#139)
- **`wraps_ufunc`** infers `output_core_dims` from the number of return units, so multi-return functions no longer need to pass it explicitly. (#141)

### Removed

- **`CompositeAlgorithm.from_list`** — renamed to `CompositeAlgorithm._build_from_list`. To build a composite from a list of names, index the registry instead: `algorithms[["a", "b"]]`. (#147)
- **`cfspopcon.formulas.__all__`** — the hand-maintained submodule list is gone, so `from cfspopcon.formulas import *` no longer pulls in the subpackages. Import them by name, or use `dir(cfspopcon.formulas)` to list them. (#147)
- **`calc_peaked_profiles`, `calc_1D_plasma_profiles` algorithms** — replaced by `calc_peaking_and_analytic_profiles` / `calc_peaking_and_prf_profiles`. (#139)
- **`density_profile_form`, `temp_profile_form` inputs** — and with them, mixed density/temperature profile forms. (#139)
