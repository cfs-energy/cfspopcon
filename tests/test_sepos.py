import numpy as np
import pytest
import xarray as xr

import cfspopcon
from cfspopcon import Quantity, ureg
from cfspopcon.formulas import separatrix_operational_space as sepos


@pytest.fixture(params=[True, False], ids=["density1D", "not-density1D"])
def density1D(request):
    return request.param


@pytest.fixture(params=[True, False], ids=["temp1D", "not-temp1D"])
def temp1D(request):
    return request.param


@pytest.fixture()
def scalar_case(density1D, temp1D):
    return not density1D and not temp1D


@pytest.fixture()
def separatrix_electron_density(density1D):
    if density1D:
        ne_sep = np.linspace(0.01, 7.0, num=40) * 1e19
        return xr.DataArray(Quantity(ne_sep, ureg.m**-3), coords=dict(dim_separatrix_electron_density=ne_sep))
    else:
        return Quantity(1.62, ureg.n19)


@pytest.fixture()
def separatrix_electron_temp(temp1D):
    if temp1D:
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
def ion_mass():
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
def ion_to_electron_temperature_ratio():
    return 0.95


@pytest.fixture()
def edge_safety_factor(
    magnetic_field_on_axis,
    major_radius,
    minor_radius,
    plasma_current,
    elongation_psi95,
    triangularity_psi95,
):
    return sepos.calc_cylindrical_edge_safety_factor(
        major_radius,
        minor_radius,
        elongation_psi95,
        triangularity_psi95,
        magnetic_field_on_axis,
        plasma_current,
    )


def test_alpha_t(
    separatrix_electron_density,
    separatrix_electron_temp,
    major_radius,
    ion_mass,
    z_effective,
    mean_ion_charge,
    ion_to_electron_temperature_ratio,
    edge_safety_factor,
    scalar_case,
):

    alpha_t = sepos.calc_alpha_t(
        electron_density=separatrix_electron_density,
        electron_temperature=separatrix_electron_temp,
        edge_safety_factor=edge_safety_factor,
        major_radius=major_radius,
        ion_mass=ion_mass,
        Zeff=z_effective,
        Z=mean_ion_charge,
        ion_to_electron_temperature_ratio=ion_to_electron_temperature_ratio,
    )

    if scalar_case:
        assert np.isclose(alpha_t, 0.5678)


def test_alpha_t_with_fixed_coulomb_log(
    separatrix_electron_density,
    separatrix_electron_temp,
    major_radius,
    ion_mass,
    z_effective,
    mean_ion_charge,
    ion_to_electron_temperature_ratio,
    edge_safety_factor,
    scalar_case,
):
    alpha_t = sepos.calc_alpha_t_with_fixed_coulomb_log(
        electron_density=separatrix_electron_density,
        electron_temperature=separatrix_electron_temp,
        edge_safety_factor=edge_safety_factor,
        major_radius=major_radius,
        ion_mass=ion_mass,
        Zeff=z_effective,
        Z=mean_ion_charge,
        ion_to_electron_temperature_ratio=ion_to_electron_temperature_ratio,
    )

    if scalar_case:
        assert np.isclose(alpha_t, 0.5928)


@pytest.fixture()
def alpha_t_turbulence_param(
    separatrix_electron_density,
    separatrix_electron_temp,
    major_radius,
    ion_mass,
    z_effective,
    mean_ion_charge,
    ion_to_electron_temperature_ratio,
    edge_safety_factor,
):
    return sepos.calc_alpha_t(
        electron_density=separatrix_electron_density,
        electron_temperature=separatrix_electron_temp,
        edge_safety_factor=edge_safety_factor,
        major_radius=major_radius,
        ion_mass=ion_mass,
        Zeff=z_effective,
        Z=mean_ion_charge,
        ion_to_electron_temperature_ratio=ion_to_electron_temperature_ratio,
    )


@pytest.fixture()
def alpha_c(elongation_psi95, triangularity_psi95):
    return sepos.calc_critical_MHD_parameter_alpha_c(elongation_psi95, triangularity_psi95)


@pytest.fixture()
def B_pol_avg(minor_radius, elongation_psi95, triangularity_psi95, plasma_current):
    poloidal_circumference = 2.0 * np.pi * minor_radius * (1 + 0.55 * (elongation_psi95 - 1)) * (1 + 0.08 * triangularity_psi95**2)
    return ureg.mu_0 * plasma_current / poloidal_circumference


@pytest.fixture()
def rho_s_pol(separatrix_electron_temp, ion_mass, B_pol_avg):

    return sepos.calc_sound_larmor_radius_rho_s(
        electron_temperature=separatrix_electron_temp, magnetic_field_strength=B_pol_avg, ion_mass=ion_mass
    )


@pytest.fixture()
def lambda_pe_H(alpha_t_turbulence_param, rho_s_pol):
    return sepos.calc_lambda_pe_Eich2021H(alpha_t_turbulence_param, rho_s_pol)


@pytest.fixture()
def lambda_pe_L(alpha_t_turbulence_param):
    return sepos.calc_lambda_pe_Manz2023L(alpha_t_turbulence_param)


def test_L_mode_density_limit_condition(
    separatrix_electron_density,
    separatrix_electron_temp,
    major_radius,
    magnetic_field_on_axis,
    ion_mass,
    alpha_c,
    alpha_t_turbulence_param,
    lambda_pe_L,
    scalar_case,
):
    L_mode_density_limit_condition = sepos.calc_L_mode_density_limit_condition(
        electron_density=separatrix_electron_density,
        electron_temperature=separatrix_electron_temp,
        major_radius=major_radius,
        magnetic_field_strength=magnetic_field_on_axis,
        ion_mass=ion_mass,
        alpha_c=alpha_c,
        alpha_t_turbulence_param=alpha_t_turbulence_param,
        lambda_pe=lambda_pe_L,
    )

    if scalar_case:
        assert np.isclose(L_mode_density_limit_condition, 0.6051721755962777)


def test_LH_transition_condition(
    separatrix_electron_density,
    separatrix_electron_temp,
    major_radius,
    magnetic_field_on_axis,
    ion_mass,
    alpha_c,
    alpha_t_turbulence_param,
    lambda_pe_H,
    scalar_case,
):
    LH_transition_condition = sepos.calc_LH_transition_condition(
        electron_density=separatrix_electron_density,
        electron_temperature=separatrix_electron_temp,
        major_radius=major_radius,
        magnetic_field_strength=magnetic_field_on_axis,
        ion_mass=ion_mass,
        alpha_c=alpha_c,
        alpha_t_turbulence_param=alpha_t_turbulence_param,
        lambda_pe=lambda_pe_H,
    )

    if scalar_case:
        assert np.isclose(LH_transition_condition, 1.0626161415333535)


@pytest.fixture()
def LH_transition_condition(
    separatrix_electron_density,
    separatrix_electron_temp,
    major_radius,
    magnetic_field_on_axis,
    ion_mass,
    alpha_c,
    alpha_t_turbulence_param,
    lambda_pe_H,
):
    return sepos.calc_LH_transition_condition(
        electron_density=separatrix_electron_density,
        electron_temperature=separatrix_electron_temp,
        major_radius=major_radius,
        magnetic_field_strength=magnetic_field_on_axis,
        ion_mass=ion_mass,
        alpha_c=alpha_c,
        alpha_t_turbulence_param=alpha_t_turbulence_param,
        lambda_pe=lambda_pe_H,
    )


def test_ideal_MHD_limit_condition(
    separatrix_electron_density,
    separatrix_electron_temp,
    major_radius,
    magnetic_field_on_axis,
    edge_safety_factor,
    alpha_c,
    alpha_t_turbulence_param,
    lambda_pe_H,
    ion_to_electron_temperature_ratio,
    scalar_case,
):
    ideal_MHD_limit_condition = sepos.calc_ideal_MHD_limit_condition(
        electron_density=separatrix_electron_density,
        electron_temperature=separatrix_electron_temp,
        major_radius=major_radius,
        magnetic_field_strength=magnetic_field_on_axis,
        safety_factor=edge_safety_factor,
        alpha_c=alpha_c,
        alpha_t_turbulence_param=alpha_t_turbulence_param,
        lambda_pe=lambda_pe_H,
        ion_to_electron_temperature_ratio=ion_to_electron_temperature_ratio,
    )

    if scalar_case:
        assert np.isclose(ideal_MHD_limit_condition, 0.3560709349613319)


def test_ideal_MHD_limit_condition_with_alpha_MHD(
    separatrix_electron_density,
    separatrix_electron_temp,
    major_radius,
    magnetic_field_on_axis,
    edge_safety_factor,
    alpha_c,
    lambda_pe_H,
    ion_to_electron_temperature_ratio,
    scalar_case,
):
    ideal_MHD_limit_condition_with_alpha_MHD = sepos.calc_ideal_MHD_limit_condition_with_alpha_MHD(
        electron_density=separatrix_electron_density,
        electron_temperature=separatrix_electron_temp,
        major_radius=major_radius,
        magnetic_field_strength=magnetic_field_on_axis,
        safety_factor=edge_safety_factor,
        alpha_c=alpha_c,
        lambda_pe=lambda_pe_H,
        ion_to_electron_temperature_ratio=ion_to_electron_temperature_ratio,
    )

    if scalar_case:
        assert np.isclose(ideal_MHD_limit_condition_with_alpha_MHD, 0.12678651072423716)


def test_LH_power_calculation(
    separatrix_electron_density,
    separatrix_electron_temp,
    alpha_t_turbulence_param,
    rho_s_pol,
    major_radius,
    minor_radius,
    edge_safety_factor,
    elongation_psi95,
    B_pol_avg,
    magnetic_field_on_axis,
    z_effective,
    LH_transition_condition,
    density1D,
    temp1D,
):
    lambda_q = sepos.calc_lambda_q_Eich2020H(alpha_t_turbulence_param, rho_s_pol)
    B_pol_omp = 16.0 / 9.0 * major_radius / (major_radius + minor_radius) * B_pol_avg
    B_tor_omp = magnetic_field_on_axis * (major_radius / (major_radius + minor_radius))

    separatrix_power = sepos.calc_power_crossing_separatrix(
        separatrix_temp=separatrix_electron_temp,
        target_temp=Quantity(10.0, ureg.eV),
        cylindrical_edge_safety_factor=edge_safety_factor,
        major_radius=major_radius,
        minor_radius=minor_radius,
        lambda_q=lambda_q,
        B_pol_omp=B_pol_omp,
        B_tor_omp=B_tor_omp,
        f_share=0.65,
        Zeff=z_effective,
    )

    # Need 2D data after this
    if not (density1D and temp1D):
        return

    LH_separatrix_density, LH_separatrix_temp = sepos.extract_LH_contour_points(LH_transition_condition)

    sepos.interpolate_field_to_LH_curve(
        separatrix_power.broadcast_like(alpha_t_turbulence_param), LH_separatrix_density, LH_separatrix_temp
    )

    kappa95_to_kappaA = 1.1  # N.b. random number
    sepos.calc_power_crossing_separatrix_in_ion_channel(
        surface_area=cfspopcon.formulas.calc_plasma_surface_area(
            major_radius, minor_radius / major_radius, elongation_psi95 * kappa95_to_kappaA
        ),
        separatrix_density=separatrix_electron_density,
        separatrix_temp=separatrix_electron_temp,
        lambda_Te=3.5 * lambda_q,
        chi_i=Quantity(0.5, ureg.m**2 / ureg.s),
        temp_scale_length_ratio=1.02,
    )
