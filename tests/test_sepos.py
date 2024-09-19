import numpy as np
import pytest
import xarray as xr

from cfspopcon import formulas
from cfspopcon.formulas.separatrix_conditions import separatrix_operational_space as sepos
from cfspopcon.unit_handling import Quantity, get_units, ureg
from cfspopcon.unit_handling import dimensionless_magnitude as dmag
from cfspopcon.unit_handling import magnitude_in_units as umag


@pytest.fixture(params=[True, False], ids=["density_scalar", "density_vector"])
def density_vector(request):
    return request.param


@pytest.fixture(params=[True, False], ids=["temp_scalar", "temp_vector"])
def temp_vector(request):
    return request.param


@pytest.fixture()
def scalar_case(density_vector, temp_vector):
    return not density_vector and not temp_vector


@pytest.fixture()
def separatrix_electron_density(density_vector):
    if density_vector:
        ne_sep = np.linspace(0.01, 7.0, num=40) * 1e19
        return xr.DataArray(Quantity(ne_sep, ureg.m**-3), coords=dict(dim_separatrix_electron_density=ne_sep))
    else:
        return Quantity(1.62, ureg.n19)


@pytest.fixture()
def separatrix_electron_temp(temp_vector):
    if temp_vector:
        te_sep = np.linspace(1, 150, num=30)
        return xr.DataArray(Quantity(te_sep, ureg.eV), coords=dict(dim_separatrix_electron_temp=te_sep))
    else:
        return Quantity(57.5, ureg.eV)


@pytest.fixture()
def magnetic_field_on_axis():
    """Toroidal field on-axis in Tesla."""
    return Quantity(2.5, ureg.T)


@pytest.fixture()
def major_radius():
    """Major radius in metres."""
    return Quantity(1.65, ureg.m)


@pytest.fixture()
def minor_radius():
    """Minor radius in metres."""
    return Quantity(0.49, ureg.m)


@pytest.fixture()
def average_ion_mass():
    """Ion mass in amu."""
    return Quantity(2.0, ureg.amu)


@pytest.fixture()
def plasma_current():
    """Plasma current in A."""
    return Quantity(0.8e6, ureg.A)


@pytest.fixture()
def elongation_psi95():
    """Elongation (kappa) at the psiN = 0.95 flux surface."""
    return 1.6


@pytest.fixture()
def triangularity_psi95():
    """Triangularity (delta) at the psiN = 0.95 flux surface."""
    return 0.3


@pytest.fixture()
def z_effective():
    """Effective ion charge."""
    return 1.25


@pytest.fixture()
def mean_ion_charge():
    """Mean ion charge."""
    return 1.1


@pytest.fixture()
def ion_to_electron_temp_ratio():
    return 0.95


@pytest.fixture()
def cylindrical_safety_factor(
    magnetic_field_on_axis,
    major_radius,
    minor_radius,
    plasma_current,
    elongation_psi95,
    triangularity_psi95,
):
    return formulas.plasma_current.safety_factor.calc_cylindrical_edge_safety_factor(
        major_radius=major_radius,
        minor_radius=minor_radius,
        elongation_psi95=elongation_psi95,
        triangularity_psi95=triangularity_psi95,
        magnetic_field_on_axis=magnetic_field_on_axis,
        plasma_current=plasma_current,
    )


@pytest.fixture()
def alpha_t(
    separatrix_electron_density,
    separatrix_electron_temp,
    major_radius,
    average_ion_mass,
    z_effective,
    mean_ion_charge,
    ion_to_electron_temp_ratio,
    cylindrical_safety_factor,
):
    return formulas.metrics.collisionality.calc_alpha_t(
        separatrix_electron_density=separatrix_electron_density,
        separatrix_electron_temp=separatrix_electron_temp,
        cylindrical_safety_factor=cylindrical_safety_factor,
        major_radius=major_radius,
        average_ion_mass=average_ion_mass,
        z_effective=z_effective,
        mean_ion_charge_state=mean_ion_charge,
        ion_to_electron_temp_ratio=ion_to_electron_temp_ratio,
    )


def test_alpha_t(
    alpha_t,
    scalar_case,
):
    if scalar_case:
        assert np.isclose(dmag(alpha_t), 0.51587362)


@pytest.fixture()
def critical_alpha_MHD(elongation_psi95, triangularity_psi95):
    return sepos.calc_critical_alpha_MHD(elongation_psi95, triangularity_psi95)


@pytest.fixture()
def B_pol_avg(minor_radius, elongation_psi95, triangularity_psi95, plasma_current):
    poloidal_circumference = 2.0 * np.pi * minor_radius * (1 + 0.55 * (elongation_psi95 - 1)) * (1 + 0.08 * triangularity_psi95**2)
    return ureg.mu_0 * plasma_current / poloidal_circumference


@pytest.fixture()
def poloidal_sound_larmor_radius(
    minor_radius,
    elongation_psi95,
    triangularity_psi95,
    plasma_current,
    separatrix_electron_temp,
    average_ion_mass,
):
    return sepos.calc_poloidal_sound_larmor_radius(
        minor_radius=minor_radius,
        elongation_psi95=elongation_psi95,
        triangularity_psi95=triangularity_psi95,
        plasma_current=plasma_current,
        separatrix_electron_temp=separatrix_electron_temp,
        average_ion_mass=average_ion_mass,
    )


@pytest.fixture()
def L_mode_density_limit_condition(
    separatrix_electron_density,
    separatrix_electron_temp,
    major_radius,
    magnetic_field_on_axis,
    average_ion_mass,
    critical_alpha_MHD,
    alpha_t,
):
    return sepos.calc_SepOS_L_mode_density_limit(
        separatrix_electron_density=separatrix_electron_density,
        separatrix_electron_temp=separatrix_electron_temp,
        major_radius=major_radius,
        magnetic_field_on_axis=magnetic_field_on_axis,
        average_ion_mass=average_ion_mass,
        critical_alpha_MHD=critical_alpha_MHD,
        alpha_t=alpha_t,
    )


def test_L_mode_density_limit_condition(
    L_mode_density_limit_condition,
    scalar_case,
):
    if scalar_case:
        assert np.isclose(dmag(L_mode_density_limit_condition), 0.5809714)


@pytest.fixture()
def LH_transition_condition(
    separatrix_electron_density,
    separatrix_electron_temp,
    major_radius,
    magnetic_field_on_axis,
    average_ion_mass,
    critical_alpha_MHD,
    alpha_t,
    poloidal_sound_larmor_radius,
):
    return sepos.calc_SepOS_LH_transition(
        separatrix_electron_density=separatrix_electron_density,
        separatrix_electron_temp=separatrix_electron_temp,
        major_radius=major_radius,
        magnetic_field_on_axis=magnetic_field_on_axis,
        average_ion_mass=average_ion_mass,
        critical_alpha_MHD=critical_alpha_MHD,
        alpha_t=alpha_t,
        poloidal_sound_larmor_radius=poloidal_sound_larmor_radius,
    )


def test_LH_transition_condition(
    LH_transition_condition,
    scalar_case,
):
    if scalar_case:
        assert np.isclose(dmag(LH_transition_condition), 1.13663158)


@pytest.fixture()
def ideal_MHD_limit_condition(
    separatrix_electron_density,
    separatrix_electron_temp,
    major_radius,
    magnetic_field_on_axis,
    cylindrical_safety_factor,
    critical_alpha_MHD,
    alpha_t,
    ion_to_electron_temp_ratio,
    poloidal_sound_larmor_radius,
):
    return sepos.calc_SepOS_ideal_MHD_limit(
        separatrix_electron_density=separatrix_electron_density,
        separatrix_electron_temp=separatrix_electron_temp,
        major_radius=major_radius,
        magnetic_field_on_axis=magnetic_field_on_axis,
        cylindrical_safety_factor=cylindrical_safety_factor,
        critical_alpha_MHD=critical_alpha_MHD,
        alpha_t=alpha_t,
        ion_to_electron_temp_ratio=ion_to_electron_temp_ratio,
        poloidal_sound_larmor_radius=poloidal_sound_larmor_radius,
    )


def test_ideal_MHD_limit_condition(ideal_MHD_limit_condition, scalar_case):
    if scalar_case:
        assert np.isclose(dmag(ideal_MHD_limit_condition), 0.37363964)


@pytest.fixture()
def ion_heat_diffusivity():
    return Quantity(0.5, ureg.m**2 / ureg.s)


@pytest.fixture()
def power_crossing_separatrix_in_ion_channel(
    surface_area,
    separatrix_electron_density,
    separatrix_electron_temp,
    alpha_t,
    poloidal_sound_larmor_radius,
    ion_heat_diffusivity,
):
    return sepos.calc_power_crossing_separatrix_in_ion_channel(
        surface_area=surface_area,
        separatrix_electron_density=separatrix_electron_density,
        separatrix_electron_temp=separatrix_electron_temp,
        alpha_t=alpha_t,
        poloidal_sound_larmor_radius=poloidal_sound_larmor_radius,
        ion_heat_diffusivity=ion_heat_diffusivity,
    )


def test_power_crossing_separatrix_in_ion_channel(
    power_crossing_separatrix_in_ion_channel,
    scalar_case,
):
    if scalar_case:
        assert np.isclose(umag(power_crossing_separatrix_in_ion_channel, ureg.MW), 0.18876955)


@pytest.fixture()
def target_electron_temp():
    return Quantity(11.2, ureg.eV)


@pytest.fixture()
def B_pol_out_mid(
    plasma_current,
    minor_radius,
):
    return formulas.scrape_off_layer.heat_flux_density.calc_B_pol_omp(
        plasma_current=plasma_current,
        minor_radius=minor_radius,
    )


@pytest.fixture()
def B_t_out_mid(
    magnetic_field_on_axis,
    major_radius,
    minor_radius,
):
    return formulas.scrape_off_layer.heat_flux_density.calc_B_tor_omp(
        magnetic_field_on_axis=magnetic_field_on_axis,
        major_radius=major_radius,
        minor_radius=minor_radius,
    )


@pytest.fixture()
def fraction_of_P_SOL_to_divertor():
    return 0.65


@pytest.fixture()
def inverse_aspect_ratio(minor_radius, major_radius):
    return minor_radius / major_radius


@pytest.fixture()
def elongation_ratio_areal_to_psi95():
    return 1.025


@pytest.fixture()
def areal_elongation(elongation_psi95, elongation_ratio_areal_to_psi95):
    return elongation_psi95 * elongation_ratio_areal_to_psi95


@pytest.fixture()
def surface_area(
    major_radius,
    inverse_aspect_ratio,
    areal_elongation,
):
    return formulas.geometry.calc_plasma_surface_area(
        major_radius=major_radius,
        inverse_aspect_ratio=inverse_aspect_ratio,
        areal_elongation=areal_elongation,
    )


@pytest.fixture()
def power_crossing_separatrix_in_electron_channel(
    separatrix_electron_temp,
    target_electron_temp,
    cylindrical_safety_factor,
    major_radius,
    minor_radius,
    B_pol_out_mid,
    B_t_out_mid,
    fraction_of_P_SOL_to_divertor,
    z_effective,
    alpha_t,
    poloidal_sound_larmor_radius,
):
    return sepos.calc_power_crossing_separatrix_in_electron_channel(
        separatrix_electron_temp=separatrix_electron_temp,
        target_electron_temp=target_electron_temp,
        cylindrical_safety_factor=cylindrical_safety_factor,
        major_radius=major_radius,
        minor_radius=minor_radius,
        B_pol_out_mid=B_pol_out_mid,
        B_t_out_mid=B_t_out_mid,
        fraction_of_P_SOL_to_divertor=fraction_of_P_SOL_to_divertor,
        z_effective=z_effective,
        alpha_t=alpha_t,
        poloidal_sound_larmor_radius=poloidal_sound_larmor_radius,
    )


def test_power_crossing_separatrix_in_electron_channel(
    power_crossing_separatrix_in_electron_channel,
    scalar_case,
):
    if scalar_case:
        assert np.isclose(umag(power_crossing_separatrix_in_electron_channel, ureg.MW), 0.67949193)


def test_interpolation_onto_LH_transition(
    LH_transition_condition,
    density_vector,
    temp_vector,
    power_crossing_separatrix_in_electron_channel,
    power_crossing_separatrix_in_ion_channel,
    separatrix_electron_density,
    separatrix_electron_temp,
):
    from cfspopcon.shaping_and_selection import find_coords_of_contour, interpolate_onto_line

    if not (density_vector and temp_vector):
        return

    contour_x, contour_y = find_coords_of_contour(
        LH_transition_condition, x_coord="dim_separatrix_electron_density", y_coord="dim_separatrix_electron_temp", level=1.0
    )

    interpolate_onto_line(power_crossing_separatrix_in_electron_channel, contour_x, contour_y)
    interpolate_onto_line(power_crossing_separatrix_in_ion_channel, contour_x, contour_y)

    temp_min = np.min(contour_y.values) * get_units(separatrix_electron_temp)
    density_min = contour_x.values[np.argmin(contour_y.values)] * get_units(separatrix_electron_density)

    assert np.isclose(umag(temp_min, ureg.eV), 53.10463924)
    assert np.isclose(umag(density_min, ureg.n19), 2.16076923)
