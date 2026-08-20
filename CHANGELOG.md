# Changelog

All notable changes to cfspopcon will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

### Added

- **`register_plugins(*packages)`** — register external plugin packages. Importing a package registers the algorithms its modules define, and a `plugin_variables.yaml` in its package root declares their default units; the names of the algorithms registered are returned. Registration is all-or-nothing: a package which fails to import, or which redefines the default units of a variable that is already defined, is rolled back along with the rest of the set.
- **A `plugins` section in an input file** — a list of importable packages, registered before the `algorithms` names are resolved, so that a case can name a plugin's algorithms and variables. Note that this imports the modules the file names.
- **`popcon_algorithms --plugin` / `-p`** (repeatable) — list a plugin's algorithms alongside the builtins.
- **`default_units_map()`** — a copy of the registered default units map, for a caller which needs to restore it.
- **[Authoring a Plugin](https://cfspopcon.readthedocs.io/en/latest/doc_sources/authoring_a_plugin.html)** documentation page, with a worked example.
- **JCH profile algorithms** — `calc_jch_profiles`, `calc_jch_pedestal_peaking`. (#139)
- **Profile-selection composite algorithms** — `calc_peaking_and_analytic_profiles`, `calc_peaking_and_prf_profiles`. (#139)
- **Radial-grid algorithm** — `define_radial_grid`, which provides `rho`. (#139)
- **Forward energy-confinement algorithms** — `calc_energy_confinement_time_from_scaling` and `calc_energy_confinement_time_from_stored_energy_and_input_power`, giving `energy_confinement_time` from a known input power. (#141)
- **`calc_H98y2`** — energy confinement time relative to the ITER98y2 scaling; adds the `H98y2` output. (#141)
- **Fixed-auxiliary-power balance** — `calc_input_power_for_fixed_auxiliary_power` and the `calc_power_balance_from_input_P_aux` composite. (#141)
- **`extend_default_units_map` is exported from `cfspopcon.unit_handling`** — one way for a downstream package to declare default units for variables of its own. It was previously reachable only from `cfspopcon.unit_handling.default_units`.

### Changed

- **`read_default_units_from_file(path)`** takes the file to read, defaulting to cfspopcon's own `variables.yaml` as before. This is how a plugin's variables file is loaded.
- **Profile form is selected by algorithm** — list a `calc_peaking_and_*_profiles` composite instead of setting the `density_profile_form` / `temp_profile_form` inputs. (#139)
- **`calc_analytic_profiles`, `calc_prf_profiles` algorithms** now take `rho` as an input and no longer return it; the `npoints` argument is removed. (#139)
- **`wraps_ufunc`** infers `output_core_dims` from the number of return units, so multi-return functions no longer need to pass it explicitly. (#141)

### Fixed

- **A named `CompositeAlgorithm`'s docstring** listed only its name. Operator precedence between the conditional expression and the implicitly concatenated strings meant the components, inputs and outputs were built only for an unnamed composite; since every registered composite is named, they were missing everywhere they were used.

### Removed

- **`calc_peaked_profiles`, `calc_1D_plasma_profiles` algorithms** — replaced by `calc_peaking_and_analytic_profiles` / `calc_peaking_and_prf_profiles`. (#139)
- **`density_profile_form`, `temp_profile_form` inputs** — and with them, mixed density/temperature profile forms. (#139)
