import numpy as np
import pytest
import xarray as xr

from cfspopcon.formulas.fusion_power import (
    DDFusionBoschHale,
    DDFusionHively,
    DTFusionBoschHale,
    calc_average_ion_mass,
    calc_fusion_power,
    calc_neutron_flux_to_walls,
)
from cfspopcon.formulas.plasma_profiles.jch_profiles import calc_jch_profiles
from cfspopcon.formulas.plasma_profiles.plasma_profiles import (
    calc_analytic_profiles,
    calc_prf_profiles,
    define_radial_grid,
)
from cfspopcon.unit_handling import Quantity, magnitude_in_units, ureg

# Bosch & Hale (1992), Nucl. Fusion 32 611, Table VIII (with 1993 erratum, Nucl. Fusion 33 1919):
# (T_i [keV], <sigma*v>_{D(d,p)T} [cm^3/s], <sigma*v>_{D(d,n)3He} [cm^3/s])
BOSCH_HALE_TABLE_VIII_DD = [
    (0.2, 4.640e-28, 4.482e-28),
    (0.3, 2.071e-26, 2.004e-26),
    (0.4, 2.237e-25, 2.168e-25),
    (0.5, 1.204e-24, 1.169e-24),
    (0.6, 4.321e-24, 4.200e-24),
    (0.7, 1.193e-23, 1.162e-23),
    (0.8, 2.751e-23, 2.681e-23),
    (1.0, 1.017e-22, 9.933e-23),
    (1.25, 3.387e-22, 3.319e-22),
    (1.3, 4.143e-22, 4.062e-22),
    (1.5, 8.431e-22, 8.284e-22),
    (1.75, 1.739e-21, 1.713e-21),
    (1.8, 1.976e-21, 1.948e-21),
    (2.0, 3.150e-21, 3.110e-21),
    (2.5, 7.969e-21, 7.905e-21),
    (3.0, 1.608e-20, 1.602e-20),
    (4.0, 4.428e-20, 4.447e-20),
    (5.0, 9.024e-20, 9.128e-20),
    (6.0, 1.545e-19, 1.573e-19),
    (8.0, 3.354e-19, 3.457e-19),
    (10.0, 5.781e-19, 6.023e-19),
    (12.0, 8.723e-19, 9.175e-19),
    (15.0, 1.390e-18, 1.481e-18),
    (20.0, 2.399e-18, 2.603e-18),
    (30.0, 4.728e-18, 5.271e-18),
    (40.0, 7.249e-18, 8.235e-18),
    (50.0, 9.838e-18, 1.133e-17),
]


@pytest.mark.parametrize(("ion_temp_keV", "expected_pT", "expected_nHe3"), BOSCH_HALE_TABLE_VIII_DD)
def test_dd_bosch_hale_rate_coefficient_matches_table_viii(ion_temp_keV: float, expected_pT: float, expected_nHe3: float) -> None:
    reaction = DDFusionBoschHale()
    sigmav_combined, sigmav_pT, sigmav_nHe3 = reaction.calc_rate_coefficient(Quantity(ion_temp_keV, ureg.keV))

    unit = ureg.cm**3 / ureg.s
    np.testing.assert_allclose(magnitude_in_units(sigmav_pT, unit), expected_pT, rtol=1e-3)
    np.testing.assert_allclose(magnitude_in_units(sigmav_nHe3, unit), expected_nHe3, rtol=1e-3)
    np.testing.assert_allclose(magnitude_in_units(sigmav_combined, unit), expected_pT + expected_nHe3, rtol=1e-3)


@pytest.mark.parametrize("ion_temp_keV", [100.1, 1000.0])
def test_dd_bosch_hale_above_validity_range_returns_nan(ion_temp_keV: float) -> None:
    reaction = DDFusionBoschHale()
    sigmav_combined, sigmav_pT, sigmav_nHe3 = reaction.calc_rate_coefficient(Quantity(ion_temp_keV, ureg.keV))

    unit = ureg.cm**3 / ureg.s
    assert np.isnan(magnitude_in_units(sigmav_combined, unit))
    assert np.isnan(magnitude_in_units(sigmav_pT, unit))
    assert np.isnan(magnitude_in_units(sigmav_nHe3, unit))


@pytest.mark.parametrize("ion_temp_keV", [-1.0, 0.0, 1e-4, 0.05])
def test_dd_bosch_hale_low_and_nonpositive_temperatures_are_finite(ion_temp_keV: float) -> None:
    reaction = DDFusionBoschHale()
    sigmav_combined, sigmav_pT, sigmav_nHe3 = reaction.calc_rate_coefficient(Quantity(ion_temp_keV, ureg.keV))

    unit = ureg.cm**3 / ureg.s
    for rate in (sigmav_combined, sigmav_pT, sigmav_nHe3):
        val = float(magnitude_in_units(rate, unit))
        assert np.isfinite(val)
        assert val >= 0.0
        if ion_temp_keV <= 0.0:
            assert val == 0.0


def test_dd_bosch_hale_energy_per_reaction_methods() -> None:
    reaction = DDFusionBoschHale()
    temps = xr.DataArray([0.0, 10.0, 20.0], dims=["dim_rho"]) * ureg.keV

    e_total = reaction.calc_energy_per_reaction(temps)
    e_neut = reaction.calc_energy_to_neutrals_per_reaction(temps)
    e_charged = reaction.calc_energy_to_charged_per_reaction(temps)

    np.testing.assert_allclose(
        magnitude_in_units(e_total, ureg.MJ),
        magnitude_in_units(e_neut + e_charged, ureg.MJ),
    )
    assert float(magnitude_in_units(e_total.isel(dim_rho=0), ureg.MJ)) == 0.0
    assert float(magnitude_in_units(e_total.isel(dim_rho=1), ureg.MeV)) == pytest.approx(3.64, rel=1e-2)


def test_dd_fusion_power_and_neutron_rate_for_uniform_profiles() -> None:
    rho = xr.DataArray(np.linspace(0.0, 1.0, 50), dims=["dim_rho"]) * ureg.dimensionless
    ion_temp = 20.0 * ureg.keV
    ion_temp_profile = xr.DataArray(np.full(50, 20.0), dims=["dim_rho"]) * ureg.keV
    fuel_ion_density = 10.0 * ureg.n19
    fuel_ion_density_profile = xr.DataArray(np.full(50, 10.0), dims=["dim_rho"]) * ureg.n19
    plasma_volume = 100.0 * ureg.m**3
    surface_area = 150.0 * ureg.m**2
    heavier_fuel_species_fraction = 0.5 * ureg.dimensionless

    P_fusion, P_neutron, P_alpha = calc_fusion_power(
        fusion_reaction="DDFusionBoschHale",
        ion_temp_profile=ion_temp_profile,
        heavier_fuel_species_fraction=heavier_fuel_species_fraction,
        fuel_ion_density_profile=fuel_ion_density_profile,
        rho=rho,
        plasma_volume=plasma_volume,
    )

    reaction = DDFusionBoschHale()
    _, sigmav_pT, sigmav_nHe3 = reaction.calc_rate_coefficient(ion_temp)
    expected_nHe3_rate = 0.5 * (fuel_ion_density**2) * sigmav_nHe3 * plasma_volume
    expected_pT_rate = 0.5 * (fuel_ion_density**2) * sigmav_pT * plasma_volume
    expected_P_neutron = expected_nHe3_rate * reaction.energy_to_neutrals_per_DD_to_nHe3_reaction
    expected_P_alpha = (
        expected_pT_rate * reaction.energy_to_charged_per_DD_to_pT_reaction
        + expected_nHe3_rate * reaction.energy_to_charged_per_DD_to_nHe3_reaction
    )

    np.testing.assert_allclose(
        magnitude_in_units(P_neutron, ureg.MW),
        magnitude_in_units(expected_P_neutron, ureg.MW),
    )
    np.testing.assert_allclose(
        magnitude_in_units(P_alpha, ureg.MW),
        magnitude_in_units(expected_P_alpha, ureg.MW),
    )
    np.testing.assert_allclose(
        magnitude_in_units(P_fusion, ureg.MW),
        magnitude_in_units(expected_P_neutron + expected_P_alpha, ureg.MW),
    )

    neutron_power_flux, neutron_rate = calc_neutron_flux_to_walls(
        P_neutron=P_neutron,
        surface_area=surface_area,
        fusion_reaction="DDFusionBoschHale",
        ion_temp_profile=ion_temp_profile,
        heavier_fuel_species_fraction=heavier_fuel_species_fraction,
    )

    np.testing.assert_allclose(
        magnitude_in_units(neutron_power_flux, ureg.MW / ureg.m**2),
        magnitude_in_units(expected_P_neutron / surface_area, ureg.MW / ureg.m**2),
    )
    np.testing.assert_allclose(
        magnitude_in_units(neutron_rate, ureg.s**-1),
        magnitude_in_units(expected_nHe3_rate, ureg.s**-1),
    )

    average_ion_mass = calc_average_ion_mass(
        fusion_reaction="DDFusionBoschHale",
        heavier_fuel_species_fraction=heavier_fuel_species_fraction,
    )
    np.testing.assert_allclose(magnitude_in_units(average_ion_mass, ureg.amu), 2.0)


@pytest.mark.parametrize("profile_kind", ["analytic", "prf", "jch"])
def test_dd_fusion_power_and_neutron_rate_across_peaked_profile_families(profile_kind: str) -> None:
    """Peaked profiles with cold edge knots (< 0.2 keV) must integrate to finite positive DD power."""
    if profile_kind == "analytic":
        rho = define_radial_grid(50) * ureg.dimensionless
        _, ni, _, Ti = calc_analytic_profiles(
            rho=rho,
            average_electron_density=15.0 * ureg.n19,
            average_electron_temp=5.0 * ureg.keV,
            average_ion_temp=5.0 * ureg.keV,
            electron_density_peaking=1.3 * ureg.dimensionless,
            ion_density_peaking=1.3 * ureg.dimensionless,
            temperature_peaking=2.5 * ureg.dimensionless,
            dilution=0.85 * ureg.dimensionless,
        )
    elif profile_kind == "prf":
        rho = define_radial_grid(50) * ureg.dimensionless
        _, ni, _, Ti = calc_prf_profiles(
            rho=rho,
            average_electron_density=15.0 * ureg.n19,
            average_electron_temp=2.5 * ureg.keV,
            average_ion_temp=2.5 * ureg.keV,
            electron_density_peaking=1.3 * ureg.dimensionless,
            ion_density_peaking=1.3 * ureg.dimensionless,
            temperature_peaking=2.5 * ureg.dimensionless,
            dilution=0.85 * ureg.dimensionless,
            normalized_inverse_temp_scale_length=2.5 * ureg.dimensionless,
        )
    else:
        rho_raw, _, ni_raw, _, Ti_raw = calc_jch_profiles(
            average_electron_density=15.0 * ureg.n19,
            average_electron_temp=5.0 * ureg.keV,
            average_ion_temp=5.0 * ureg.keV,
            electron_density_peaking=1.3 * ureg.dimensionless,
            ion_density_peaking=1.3 * ureg.dimensionless,
            temperature_peaking=2.5 * ureg.dimensionless,
            dilution=0.85 * ureg.dimensionless,
            n_points_for_confined_region_profiles=100,
            separatrix_electron_temp=0.10 * ureg.keV,
        )
        rho = rho_raw * ureg.dimensionless
        ni = xr.DataArray(ni_raw, coords=dict(dim_rho=rho))
        Ti = xr.DataArray(Ti_raw, coords=dict(dim_rho=rho))

    # Edge temperature knot is below the 0.2 keV Bosch-Hale table minimum
    assert float(magnitude_in_units(Ti.isel(dim_rho=-1), ureg.keV)) < 0.2

    P_fusion, P_neutron, P_alpha = calc_fusion_power(
        fusion_reaction="DDFusionBoschHale",
        ion_temp_profile=Ti,
        heavier_fuel_species_fraction=0.5 * ureg.dimensionless,
        fuel_ion_density_profile=ni,
        rho=rho,
        plasma_volume=20.0 * ureg.m**3,
    )

    for power in (P_fusion, P_neutron, P_alpha):
        val = float(magnitude_in_units(power, ureg.MW))
        assert np.isfinite(val)
        assert val > 0.0

    np.testing.assert_allclose(
        magnitude_in_units(P_fusion, ureg.MW),
        magnitude_in_units(P_neutron + P_alpha, ureg.MW),
    )

    _, neutron_rate = calc_neutron_flux_to_walls(
        P_neutron=P_neutron,
        surface_area=60.0 * ureg.m**2,
        fusion_reaction="DDFusionBoschHale",
        ion_temp_profile=Ti,
        heavier_fuel_species_fraction=0.5 * ureg.dimensionless,
    )
    np.testing.assert_allclose(
        magnitude_in_units(neutron_rate * (2.45 * ureg.MeV), ureg.MW),
        magnitude_in_units(P_neutron, ureg.MW),
    )


def test_disabled_reactions_still_raise() -> None:
    rho = xr.DataArray(np.linspace(0.0, 1.0, 10), dims=["dim_rho"]) * ureg.dimensionless
    ion_temp_profile = xr.DataArray(np.full(10, 10.0), dims=["dim_rho"]) * ureg.keV
    fuel_ion_density_profile = xr.DataArray(np.full(10, 10.0), dims=["dim_rho"]) * ureg.n19

    assert isinstance(DDFusionHively(), DDFusionBoschHale)
    assert not isinstance(DTFusionBoschHale(), DDFusionBoschHale)

    with pytest.raises(NotImplementedError, match="currently disabled"):
        calc_fusion_power(
            fusion_reaction="DDFusionHively",
            ion_temp_profile=ion_temp_profile,
            heavier_fuel_species_fraction=0.5 * ureg.dimensionless,
            fuel_ion_density_profile=fuel_ion_density_profile,
            rho=rho,
            plasma_volume=100.0 * ureg.m**3,
        )
