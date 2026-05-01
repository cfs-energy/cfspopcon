import numpy as np

from cfspopcon.formulas.plasma_profiles import (
    calc_analytic_profiles,
    calc_jch_pedestal_peaking,
    calc_jch_profiles,
    define_radial_grid,
)
from cfspopcon.unit_handling import magnitude_in_units, ureg


def expected_edge_nudge(npoints: int) -> float:
    return 0.1 / (npoints - 1 + 0.1)


def test_calc_1d_plasma_profiles_jch_respects_requested_peaking():
    rho, electron_density, fuel_ion_density, electron_temp, ion_temp = calc_jch_profiles(
        average_electron_density=20.0 * ureg.n19,
        average_electron_temp=10.0 * ureg.keV,
        average_ion_temp=12.0 * ureg.keV,
        electron_density_peaking=1.5,
        ion_density_peaking=1.2,
        temperature_peaking=2.0,
        dilution=0.8,
        pedestal_width=0.05,
        t_sep=0.2 * ureg.keV,
        n_sep_ratio=0.5,
    )

    rho_mag = magnitude_in_units(rho, ureg.dimensionless)
    electron_density_mag = magnitude_in_units(electron_density, ureg.n19)
    fuel_ion_density_mag = magnitude_in_units(fuel_ion_density, ureg.n19)
    electron_temp_mag = magnitude_in_units(electron_temp, ureg.keV)
    ion_temp_mag = magnitude_in_units(ion_temp, ureg.keV)

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
    rho, *_ = calc_jch_profiles(
        average_electron_density=20.0 * ureg.n19,
        average_electron_temp=10.0 * ureg.keV,
        average_ion_temp=12.0 * ureg.keV,
        electron_density_peaking=1.5,
        ion_density_peaking=1.2,
        temperature_peaking=2.0,
        dilution=0.8,
        pedestal_width=0.05,
        t_sep=0.2 * ureg.keV,
        n_sep_ratio=0.5,
    )

    rho_mag = np.asarray(magnitude_in_units(rho, ureg.dimensionless))

    assert rho_mag.size == 50
    np.testing.assert_allclose(rho_mag[-4:], np.linspace(0.95, 1.0 - expected_edge_nudge(len(rho_mag)), 4), rtol=0.0, atol=1e-12)


def test_calc_1d_plasma_profiles_default_grid_stops_just_short_of_separatrix():
    rho, *_ = calc_jch_profiles(
        average_electron_density=20.0 * ureg.n19,
        average_electron_temp=10.0 * ureg.keV,
        average_ion_temp=12.0 * ureg.keV,
        electron_density_peaking=1.5,
        ion_density_peaking=1.2,
        temperature_peaking=2.0,
        dilution=0.8,
    )

    rho_mag = np.asarray(magnitude_in_units(rho, ureg.dimensionless))
    edge_nudge = expected_edge_nudge(len(rho_mag))

    np.testing.assert_allclose(rho_mag[[0, -1]], [0.0, 1.0 - edge_nudge], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(np.trapezoid(np.ones_like(rho_mag) * 2.0 * rho_mag, x=rho_mag), rho_mag[-1] ** 2, rtol=1e-12)




def test_calc_1d_plasma_profiles_jch_small_pedestal_keeps_four_edge_points():
    rho, electron_density, _, electron_temp, _ = calc_jch_profiles(
        average_electron_density=20.0 * ureg.n19,
        average_electron_temp=10.0 * ureg.keV,
        average_ion_temp=12.0 * ureg.keV,
        electron_density_peaking=1.5,
        ion_density_peaking=1.2,
        temperature_peaking=2.0,
        dilution=0.8,
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

    rho = define_radial_grid()

    electron_density, fuel_ion_density, electron_temp, ion_temp = calc_analytic_profiles(
        rho = rho,
        average_electron_density=20.0 * ureg.n19,
        average_electron_temp=0.1 * ureg.keV,
        average_ion_temp=0.1 * ureg.keV,
        electron_density_peaking=1.5,
        ion_density_peaking=1.2,
        temperature_peaking=1.1,
        dilution=0.8,
    )

    assert rho.shape == electron_density.shape == fuel_ion_density.shape == electron_temp.shape == ion_temp.shape


def test_calc_peaked_profiles_jch_reports_volume_and_pedestal_peaking():

    from cfspopcon.formulas.plasma_profiles import (
        calc_effective_collisionality,
        calc_electron_density_peaking,
        calc_ion_density_peaking,
        calc_peak_electron_temp,
        calc_peak_ion_temp,
    )

    average_electron_density=20.0 * ureg.n19
    average_electron_temp=10.0 * ureg.keV
    average_ion_temp=12.0 * ureg.keV
    ion_density_peaking_offset=0.0 * ureg.dimensionless
    electron_density_peaking_offset=0.0 * ureg.dimensionless
    temperature_peaking=2.0 * ureg.dimensionless
    major_radius=1.85 * ureg.m
    z_effective=1.5 * ureg.dimensionless
    dilution=0.8 * ureg.dimensionless
    beta_toroidal=0.02 * ureg.dimensionless
    pedestal_width=0.05 * ureg.dimensionless
    t_sep=0.2 * ureg.keV
    n_sep_ratio=0.5 * ureg.dimensionless
    n_points_for_confined_region_profiles = 50

    effective_collisionality = calc_effective_collisionality(
        average_electron_density = average_electron_density,
        average_electron_temp = average_electron_temp,
        major_radius = major_radius,
        z_effective = z_effective,
    )

    electron_density_peaking, peak_electron_density = calc_electron_density_peaking(
        effective_collisionality = effective_collisionality,
        beta_toroidal = beta_toroidal,
        electron_density_peaking_offset = electron_density_peaking_offset,
        average_electron_density = average_electron_density,
    )

    ion_density_peaking, peak_fuel_ion_density = calc_ion_density_peaking(
        effective_collisionality = effective_collisionality,
        beta_toroidal = beta_toroidal,
        ion_density_peaking_offset = ion_density_peaking_offset,
        average_electron_density = average_electron_density,
        dilution = dilution,
    )

    rho, electron_density_profile, fuel_ion_density_profile, electron_temp_profile, ion_temp_profile = calc_jch_profiles(
        average_electron_density=average_electron_density,
        average_electron_temp=average_electron_temp,
        average_ion_temp=average_ion_temp,
        electron_density_peaking=electron_density_peaking,
        ion_density_peaking=ion_density_peaking,
        temperature_peaking=temperature_peaking,
        dilution=dilution,
        pedestal_width=pedestal_width,
        t_sep=t_sep,
        n_sep_ratio=n_sep_ratio,
        n_points_for_confined_region_profiles = n_points_for_confined_region_profiles,
    )

    peak_electron_temp = calc_peak_electron_temp(average_electron_temp=average_electron_temp, temperature_peaking=temperature_peaking)
    peak_ion_temp = calc_peak_ion_temp(average_ion_temp=average_ion_temp, temperature_peaking=temperature_peaking)

    electron_density_pedestal_peaking, ion_density_pedestal_peaking, electron_temp_pedestal_peaking, ion_temp_pedestal_peaking = calc_jch_pedestal_peaking(
        average_electron_temp = average_electron_temp,
        average_ion_temp = average_ion_temp,
        electron_density_peaking = electron_density_peaking,
        ion_density_peaking = ion_density_peaking,
        temperature_peaking = temperature_peaking,
        n_points_for_confined_region_profiles = n_points_for_confined_region_profiles,
        pedestal_width = pedestal_width,
        t_sep = t_sep,
        n_sep_ratio = n_sep_ratio,
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
    np.testing.assert_allclose(
        magnitude_in_units(electron_density_pedestal_peaking, ureg.dimensionless),
        electron_density_mag[0] / electron_density_mag[knee_index],
        rtol=1e-8,
    )
    np.testing.assert_allclose(
        magnitude_in_units(ion_density_pedestal_peaking, ureg.dimensionless),
        fuel_ion_density_mag[0] / fuel_ion_density_mag[knee_index],
        rtol=1e-8,
    )
    np.testing.assert_allclose(
        magnitude_in_units(electron_temp_pedestal_peaking, ureg.dimensionless),
        electron_temp_mag[0] / electron_temp_mag[knee_index],
        rtol=1e-8,
    )
    np.testing.assert_allclose(
        magnitude_in_units(ion_temp_pedestal_peaking, ureg.dimensionless), ion_temp_mag[0] / ion_temp_mag[knee_index], rtol=1e-8
    )
