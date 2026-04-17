import numpy as np

from cfspopcon.formulas.plasma_profiles import calc_1D_plasma_profiles, calc_peaked_profiles
from cfspopcon.named_options import ProfileForm
from cfspopcon.unit_handling import magnitude_in_units, ureg


def expected_edge_nudge(npoints: int) -> float:
    return 0.1 / (npoints - 1 + 0.1)


def test_calc_1d_plasma_profiles_jch_respects_requested_peaking():
    rho, electron_density, fuel_ion_density, electron_temp, ion_temp = calc_1D_plasma_profiles(
        density_profile_form=ProfileForm.jch,
        temp_profile_form=ProfileForm.jch,
        average_electron_density=20.0 * ureg.n19,
        average_electron_temp=10.0 * ureg.keV,
        average_ion_temp=12.0 * ureg.keV,
        electron_density_peaking=1.5,
        ion_density_peaking=1.2,
        temperature_peaking=2.0,
        dilution=0.8,
        normalized_inverse_temp_scale_length=2.5,
        pedestal_width=0.05,
        t_sep=0.2 * ureg.keV,
        n_sep_ratio=0.5,
    )

    rho_mag = np.asarray(magnitude_in_units(rho, ureg.dimensionless))
    electron_density_mag = np.asarray(magnitude_in_units(electron_density, ureg.n19))
    fuel_ion_density_mag = np.asarray(magnitude_in_units(fuel_ion_density, ureg.n19))
    electron_temp_mag = np.asarray(magnitude_in_units(electron_temp, ureg.keV))
    ion_temp_mag = np.asarray(magnitude_in_units(ion_temp, ureg.keV))

    assert rho_mag[0] == 0.0
    np.testing.assert_allclose(rho_mag[-1], 1.0 - expected_edge_nudge(len(rho_mag)), rtol=0.0, atol=1e-12)
    knee_index = np.where(np.isclose(rho_mag, 0.95))[0][0]

    np.testing.assert_allclose(electron_density_mag[0] / 20.0, 1.5, rtol=1e-6)
    np.testing.assert_allclose(fuel_ion_density_mag[0] / 16.0, 1.2, rtol=1e-6)
    np.testing.assert_allclose(electron_temp_mag[0] / 10.0, 2.0, rtol=1e-6)
    np.testing.assert_allclose(ion_temp_mag[0] / 12.0, 2.0, rtol=1e-6)
    assert electron_density_mag[0] / electron_density_mag[knee_index] > 1.5
    assert fuel_ion_density_mag[0] / fuel_ion_density_mag[knee_index] > 1.2
    assert electron_temp_mag[0] / electron_temp_mag[knee_index] > 2.0
    assert ion_temp_mag[0] / ion_temp_mag[knee_index] > 2.0
    np.testing.assert_allclose(np.trapezoid(electron_density_mag * 2.0 * rho_mag, x=rho_mag), 20.0, rtol=1e-8)
    np.testing.assert_allclose(np.trapezoid(fuel_ion_density_mag * 2.0 * rho_mag, x=rho_mag), 16.0, rtol=1e-8)
    np.testing.assert_allclose(np.trapezoid(electron_temp_mag * 2.0 * rho_mag, x=rho_mag), 10.0, rtol=1e-8)
    np.testing.assert_allclose(np.trapezoid(ion_temp_mag * 2.0 * rho_mag, x=rho_mag), 12.0, rtol=1e-8)


def test_calc_1d_plasma_profiles_jch_uses_four_pedestal_points_without_growing_grid():
    rho, *_ = calc_1D_plasma_profiles(
        density_profile_form=ProfileForm.jch,
        temp_profile_form=ProfileForm.jch,
        average_electron_density=20.0 * ureg.n19,
        average_electron_temp=10.0 * ureg.keV,
        average_ion_temp=12.0 * ureg.keV,
        electron_density_peaking=1.5,
        ion_density_peaking=1.2,
        temperature_peaking=2.0,
        dilution=0.8,
        normalized_inverse_temp_scale_length=2.5,
        pedestal_width=0.05,
        t_sep=0.2 * ureg.keV,
        n_sep_ratio=0.5,
    )

    rho_mag = np.asarray(magnitude_in_units(rho, ureg.dimensionless))

    assert rho_mag.size == 50
    np.testing.assert_allclose(rho_mag[-4:], np.linspace(0.95, 1.0 - expected_edge_nudge(len(rho_mag)), 4), rtol=0.0, atol=1e-12)


def test_calc_1d_plasma_profiles_default_grid_stops_just_short_of_separatrix():
    rho, *_ = calc_1D_plasma_profiles(
        density_profile_form=ProfileForm.analytic,
        temp_profile_form=ProfileForm.analytic,
        average_electron_density=20.0 * ureg.n19,
        average_electron_temp=10.0 * ureg.keV,
        average_ion_temp=12.0 * ureg.keV,
        electron_density_peaking=1.5,
        ion_density_peaking=1.2,
        temperature_peaking=2.0,
        dilution=0.8,
        normalized_inverse_temp_scale_length=2.5,
    )

    rho_mag = np.asarray(magnitude_in_units(rho, ureg.dimensionless))
    edge_nudge = expected_edge_nudge(len(rho_mag))

    np.testing.assert_allclose(rho_mag[[0, -1]], [0.0, 1.0 - edge_nudge], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(np.trapezoid(np.ones_like(rho_mag) * 2.0 * rho_mag, x=rho_mag), rho_mag[-1] ** 2, rtol=1e-12)


def test_calc_1d_plasma_profiles_hollow_analytic_profiles_stay_finite_at_edge():
    rho, electron_density, fuel_ion_density, electron_temp, ion_temp = calc_1D_plasma_profiles(
        density_profile_form=ProfileForm.analytic,
        temp_profile_form=ProfileForm.analytic,
        average_electron_density=20.0 * ureg.n19,
        average_electron_temp=10.0 * ureg.keV,
        average_ion_temp=12.0 * ureg.keV,
        electron_density_peaking=0.8,
        ion_density_peaking=0.7,
        temperature_peaking=0.6,
        dilution=0.8,
        normalized_inverse_temp_scale_length=2.5,
    )

    rho_mag = np.asarray(magnitude_in_units(rho, ureg.dimensionless))
    electron_density_mag = np.asarray(magnitude_in_units(electron_density, ureg.n19))
    fuel_ion_density_mag = np.asarray(magnitude_in_units(fuel_ion_density, ureg.n19))
    electron_temp_mag = np.asarray(magnitude_in_units(electron_temp, ureg.keV))
    ion_temp_mag = np.asarray(magnitude_in_units(ion_temp, ureg.keV))
    edge_nudge = expected_edge_nudge(len(rho_mag))

    np.testing.assert_allclose(rho_mag[-1], 1.0 - edge_nudge, rtol=0.0, atol=1e-12)
    assert np.isfinite(electron_density_mag).all()
    assert np.isfinite(fuel_ion_density_mag).all()
    assert np.isfinite(electron_temp_mag).all()
    assert np.isfinite(ion_temp_mag).all()
    np.testing.assert_allclose(np.trapezoid(electron_density_mag * 2.0 * rho_mag, x=rho_mag), 20.0, rtol=5e-3)
    np.testing.assert_allclose(np.trapezoid(fuel_ion_density_mag * 2.0 * rho_mag, x=rho_mag), 16.0, rtol=5e-3)
    np.testing.assert_allclose(np.trapezoid(electron_temp_mag * 2.0 * rho_mag, x=rho_mag), 10.0, rtol=5e-3)
    np.testing.assert_allclose(np.trapezoid(ion_temp_mag * 2.0 * rho_mag, x=rho_mag), 12.0, rtol=5e-3)


def test_calc_1d_plasma_profiles_jch_small_pedestal_keeps_four_edge_points():
    rho, electron_density, _, electron_temp, _ = calc_1D_plasma_profiles(
        density_profile_form=ProfileForm.jch,
        temp_profile_form=ProfileForm.jch,
        average_electron_density=20.0 * ureg.n19,
        average_electron_temp=10.0 * ureg.keV,
        average_ion_temp=12.0 * ureg.keV,
        electron_density_peaking=1.5,
        ion_density_peaking=1.2,
        temperature_peaking=2.0,
        dilution=0.8,
        normalized_inverse_temp_scale_length=2.5,
        pedestal_width=0.01,
        t_sep=0.2 * ureg.keV,
        n_sep_ratio=0.5,
    )

    rho_mag = np.asarray(magnitude_in_units(rho, ureg.dimensionless))
    electron_density_mag = np.asarray(magnitude_in_units(electron_density, ureg.n19))
    electron_temp_mag = np.asarray(magnitude_in_units(electron_temp, ureg.keV))
    edge_nudge = expected_edge_nudge(len(rho_mag))
    edge_basis_1 = edge_nudge / 0.01
    edge_basis_2 = 1.0 - edge_basis_1

    knee_index = np.where(np.isclose(rho_mag, 0.99))[0][0]

    np.testing.assert_allclose(rho_mag[-1], 1.0 - edge_nudge, rtol=0.0, atol=1e-12)
    assert knee_index < len(rho_mag) - 1
    np.testing.assert_allclose(rho_mag[-4:], np.linspace(0.99, 1.0 - edge_nudge, 4), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(electron_density_mag[-1] / electron_density_mag[knee_index], edge_basis_1 + 0.5 * edge_basis_2, rtol=1e-6)
    np.testing.assert_allclose(
        electron_temp_mag[-1],
        electron_temp_mag[knee_index] * edge_basis_1 + 0.2 * edge_basis_2,
        rtol=1e-6,
    )


def test_calc_1d_plasma_profiles_skips_jch_when_not_requested():
    rho, electron_density, fuel_ion_density, electron_temp, ion_temp = calc_1D_plasma_profiles(
        density_profile_form=ProfileForm.analytic,
        temp_profile_form=ProfileForm.analytic,
        average_electron_density=20.0 * ureg.n19,
        average_electron_temp=0.1 * ureg.keV,
        average_ion_temp=0.1 * ureg.keV,
        electron_density_peaking=1.5,
        ion_density_peaking=1.2,
        temperature_peaking=1.1,
        dilution=0.8,
        normalized_inverse_temp_scale_length=2.5,
    )

    assert rho.shape == electron_density.shape == fuel_ion_density.shape == electron_temp.shape == ion_temp.shape


def test_calc_1d_plasma_profiles_only_builds_requested_jch_branch():
    rho, electron_density, fuel_ion_density, electron_temp, ion_temp = calc_1D_plasma_profiles(
        density_profile_form=ProfileForm.jch,
        temp_profile_form=ProfileForm.analytic,
        average_electron_density=20.0 * ureg.n19,
        average_electron_temp=0.1 * ureg.keV,
        average_ion_temp=0.1 * ureg.keV,
        electron_density_peaking=1.5,
        ion_density_peaking=1.2,
        temperature_peaking=1.1,
        dilution=0.8,
        normalized_inverse_temp_scale_length=2.5,
        pedestal_width=0.05,
        t_sep=0.2 * ureg.keV,
        n_sep_ratio=0.5,
    )

    assert rho.shape == electron_density.shape == fuel_ion_density.shape == electron_temp.shape == ion_temp.shape


def test_calc_1d_plasma_profiles_mixed_jch_grid_keeps_hollow_analytic_branch_finite():
    rho, electron_density, fuel_ion_density, electron_temp, ion_temp = calc_1D_plasma_profiles(
        density_profile_form=ProfileForm.analytic,
        temp_profile_form=ProfileForm.jch,
        average_electron_density=20.0 * ureg.n19,
        average_electron_temp=10.0 * ureg.keV,
        average_ion_temp=12.0 * ureg.keV,
        electron_density_peaking=0.8,
        ion_density_peaking=0.7,
        temperature_peaking=1.5,
        dilution=0.8,
        normalized_inverse_temp_scale_length=2.5,
        pedestal_width=0.05,
        t_sep=0.2 * ureg.keV,
        n_sep_ratio=0.5,
    )

    rho_mag = np.asarray(magnitude_in_units(rho, ureg.dimensionless))
    electron_density_mag = np.asarray(magnitude_in_units(electron_density, ureg.n19))
    fuel_ion_density_mag = np.asarray(magnitude_in_units(fuel_ion_density, ureg.n19))
    electron_temp_mag = np.asarray(magnitude_in_units(electron_temp, ureg.keV))
    ion_temp_mag = np.asarray(magnitude_in_units(ion_temp, ureg.keV))

    np.testing.assert_allclose(rho_mag[-1], 1.0 - expected_edge_nudge(len(rho_mag)), rtol=0.0, atol=1e-12)
    assert np.isfinite(electron_density_mag).all()
    assert np.isfinite(fuel_ion_density_mag).all()
    assert np.isfinite(electron_temp_mag).all()
    assert np.isfinite(ion_temp_mag).all()


def test_calc_1d_plasma_profiles_mixed_jch_grid_preserves_prf_volume_average():
    rho, electron_density, fuel_ion_density, electron_temp, ion_temp = calc_1D_plasma_profiles(
        density_profile_form=ProfileForm.prf,
        temp_profile_form=ProfileForm.jch,
        average_electron_density=20.0 * ureg.n19,
        average_electron_temp=10.0 * ureg.keV,
        average_ion_temp=12.0 * ureg.keV,
        electron_density_peaking=1.5,
        ion_density_peaking=1.2,
        temperature_peaking=2.0,
        dilution=0.8,
        normalized_inverse_temp_scale_length=2.5,
        pedestal_width=0.2,
        t_sep=0.2 * ureg.keV,
        n_sep_ratio=0.5,
    )

    rho_mag = np.asarray(magnitude_in_units(rho, ureg.dimensionless))
    electron_density_mag = np.asarray(magnitude_in_units(electron_density, ureg.n19))
    fuel_ion_density_mag = np.asarray(magnitude_in_units(fuel_ion_density, ureg.n19))
    electron_temp_mag = np.asarray(magnitude_in_units(electron_temp, ureg.keV))
    ion_temp_mag = np.asarray(magnitude_in_units(ion_temp, ureg.keV))

    np.testing.assert_allclose(rho_mag[-4:], np.linspace(0.8, 1.0 - expected_edge_nudge(len(rho_mag)), 4), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(np.trapezoid(electron_density_mag * 2.0 * rho_mag, x=rho_mag), 20.0, rtol=1e-8)
    np.testing.assert_allclose(np.trapezoid(fuel_ion_density_mag * 2.0 * rho_mag, x=rho_mag), 16.0, rtol=1e-8)
    np.testing.assert_allclose(np.trapezoid(electron_temp_mag * 2.0 * rho_mag, x=rho_mag), 10.0, rtol=1e-8)
    np.testing.assert_allclose(np.trapezoid(ion_temp_mag * 2.0 * rho_mag, x=rho_mag), 12.0, rtol=1e-8)


def test_calc_peaked_profiles_jch_reports_volume_and_pedestal_peaking():
    (
        _,
        ion_density_peaking,
        electron_density_peaking,
        peak_electron_density,
        peak_fuel_ion_density,
        peak_electron_temp,
        peak_ion_temp,
        rho,
        electron_density_profile,
        fuel_ion_density_profile,
        electron_temp_profile,
        ion_temp_profile,
        ion_density_pedestal_peaking,
        electron_density_pedestal_peaking,
        electron_temp_pedestal_peaking,
        ion_temp_pedestal_peaking,
    ) = calc_peaked_profiles(
        average_electron_density=20.0 * ureg.n19,
        average_electron_temp=10.0 * ureg.keV,
        average_ion_temp=12.0 * ureg.keV,
        ion_density_peaking_offset=0.0 * ureg.dimensionless,
        electron_density_peaking_offset=0.0 * ureg.dimensionless,
        temperature_peaking=2.0 * ureg.dimensionless,
        major_radius=1.85 * ureg.m,
        z_effective=1.5 * ureg.dimensionless,
        dilution=0.8 * ureg.dimensionless,
        beta_toroidal=0.02 * ureg.dimensionless,
        normalized_inverse_temp_scale_length=2.5 * ureg.dimensionless,
        density_profile_form=ProfileForm.jch,
        temp_profile_form=ProfileForm.jch,
        pedestal_width=0.05 * ureg.dimensionless,
        t_sep=0.2 * ureg.keV,
        n_sep_ratio=0.5 * ureg.dimensionless,
    )

    rho_mag = np.asarray(magnitude_in_units(rho, ureg.dimensionless))
    electron_density_mag = np.asarray(magnitude_in_units(electron_density_profile, ureg.n19))
    fuel_ion_density_mag = np.asarray(magnitude_in_units(fuel_ion_density_profile, ureg.n19))
    electron_temp_mag = np.asarray(magnitude_in_units(electron_temp_profile, ureg.keV))
    ion_temp_mag = np.asarray(magnitude_in_units(ion_temp_profile, ureg.keV))
    knee_index = np.where(np.isclose(rho_mag, 0.95))[0][0]

    np.testing.assert_allclose(magnitude_in_units(peak_electron_density, ureg.n19), electron_density_mag[0], rtol=1e-8)
    np.testing.assert_allclose(magnitude_in_units(peak_fuel_ion_density, ureg.n19), fuel_ion_density_mag[0], rtol=1e-8)
    np.testing.assert_allclose(magnitude_in_units(peak_electron_temp, ureg.keV), electron_temp_mag[0], rtol=1e-8)
    np.testing.assert_allclose(magnitude_in_units(peak_ion_temp, ureg.keV), ion_temp_mag[0], rtol=1e-8)

    np.testing.assert_allclose(magnitude_in_units(electron_density_peaking, ureg.dimensionless), electron_density_mag[0] / 20.0, rtol=1e-6)
    np.testing.assert_allclose(magnitude_in_units(ion_density_peaking, ureg.dimensionless), fuel_ion_density_mag[0] / 16.0, rtol=1e-6)
    np.testing.assert_allclose(magnitude_in_units(electron_density_pedestal_peaking, ureg.dimensionless), electron_density_mag[0] / electron_density_mag[knee_index], rtol=1e-8)
    np.testing.assert_allclose(magnitude_in_units(ion_density_pedestal_peaking, ureg.dimensionless), fuel_ion_density_mag[0] / fuel_ion_density_mag[knee_index], rtol=1e-8)
    np.testing.assert_allclose(magnitude_in_units(electron_temp_pedestal_peaking, ureg.dimensionless), electron_temp_mag[0] / electron_temp_mag[knee_index], rtol=1e-8)
    np.testing.assert_allclose(magnitude_in_units(ion_temp_pedestal_peaking, ureg.dimensionless), ion_temp_mag[0] / ion_temp_mag[knee_index], rtol=1e-8)
