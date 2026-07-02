# Changelog

All notable changes to cfspopcon will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

### Added

- **JCH profile algorithms** — `calc_jch_profiles`, `calc_jch_pedestal_peaking`. (#139)
- **Profile-selection composite algorithms** — `calc_peaking_and_analytic_profiles`, `calc_peaking_and_prf_profiles`. (#139)
- **Radial-grid algorithm** — `define_radial_grid`, which provides `rho`. (#139)
- **Forward energy-confinement algorithms** — `calc_energy_confinement_time_from_scaling` and `calc_energy_confinement_time_from_stored_energy_and_input_power`, giving `energy_confinement_time` from a known input power. (#141)
- **`calc_H98y2`** — energy confinement time relative to the ITER98y2 scaling; adds the `H98y2` output. (#141)
- **Fixed-auxiliary-power balance** — `calc_input_power_for_fixed_auxiliary_power` and the `calc_power_balance_from_input_P_aux` composite. (#141)

### Changed

- **Profile form is selected by algorithm** — list a `calc_peaking_and_*_profiles` composite instead of setting the `density_profile_form` / `temp_profile_form` inputs. (#139)
- **`calc_analytic_profiles`, `calc_prf_profiles` algorithms** now take `rho` as an input and no longer return it; the `npoints` argument is removed. (#139)
- **`wraps_ufunc`** infers `output_core_dims` from the number of return units, so multi-return functions no longer need to pass it explicitly. (#141)

### Removed

- **`calc_peaked_profiles`, `calc_1D_plasma_profiles` algorithms** — replaced by `calc_peaking_and_analytic_profiles` / `calc_peaking_and_prf_profiles`. (#139)
- **`density_profile_form`, `temp_profile_form` inputs** — and with them, mixed density/temperature profile forms. (#139)
