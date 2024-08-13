import numpy as np
import pytest
import xarray as xr

from cfspopcon.unit_handling import Quantity, ureg, get_units
from cfspopcon.unit_handling import dimensionless_magnitude as dmag
from cfspopcon.unit_handling import magnitude_in_units as umag
from cfspopcon import formulas


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
def elongation_ratio_areal_to_psi95():
    return 1.025


@pytest.fixture()
def triangularity_psi95():
    """Triangularity (delta) at the psiN = 0.95 flux surface."""
    return 0.3


@pytest.fixture()
def z_effective():
    """Effective ion charge."""
    return 1.25


@pytest.fixture()
def ion_to_electron_temp_ratio():
    return 0.95


@pytest.fixture()
def target_electron_temp():
    return Quantity(10.0, ureg.eV)


@pytest.fixture()
def target_electron_density():
    return Quantity(152.1067012, ureg.n19)


@pytest.fixture()
def fraction_of_P_SOL_to_divertor():
    return 1.0


@pytest.fixture()
def ion_heat_diffusivity():
    return 0.5


@pytest.fixture()
def separatrix_electron_density():
    return Quantity(10, ureg.n19)


@pytest.fixture()
def power_crossing_separatrix():
    return Quantity(29.0, ureg.MW)


@pytest.fixture()
def lambda_q():
    return Quantity(0.3, ureg.mm)


@pytest.fixture()
def target_gaussian_spreading():
    return Quantity(0.15, ureg.mm)


@pytest.fixture()
def B_pol_out_mid():
    return Quantity(2.5, ureg.T)


@pytest.fixture()
def fieldline_pitch_at_omp():
    return 3.92


@pytest.fixture()
def parallel_connection_length():
    return Quantity(13.4, ureg.m)


@pytest.fixture()
def toroidal_flux_expansion():
    return 2.0


@pytest.fixture()
def kappa_e0():
    return Quantity(2400, "watt / electron_volt ** 3.5 / meter")


@pytest.fixture()
def SOL_momentum_loss_function():
    from cfspopcon.named_options import MomentumLossFunction

    return MomentumLossFunction.KotovReiter


@pytest.fixture()
def target_angle_of_incidence():
    return Quantity(2.0, ureg.degree)


@pytest.fixture()
def kappa_ez():
    return 4


@pytest.fixture()
def sheath_heat_transmission_factor():
    return 7


@pytest.fixture()
def neutral_flux_density_factor():
    return Quantity(1.5, "1 / meter ** 2 / pascal / second")


@pytest.fixture()
def separatrix_power_transient():
    return Quantity(5.0, ureg.MW)


@pytest.fixture()
def SOL_power_loss_fraction():
    return 0.9597046482


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
def q_parallel(
    power_crossing_separatrix,
    fraction_of_P_SOL_to_divertor,
    major_radius,
    minor_radius,
    lambda_q,
    fieldline_pitch_at_omp,
):
    return formulas.scrape_off_layer.heat_flux_density.calc_parallel_heat_flux_density(
        power_crossing_separatrix=power_crossing_separatrix,
        fraction_of_P_SOL_to_divertor=fraction_of_P_SOL_to_divertor,
        major_radius=major_radius,
        minor_radius=minor_radius,
        lambda_q=lambda_q,
        fieldline_pitch_at_omp=fieldline_pitch_at_omp,
    )


@pytest.fixture()
def two_point_model_fixed_tet(
    target_electron_temp,
    q_parallel,
    parallel_connection_length,
    separatrix_electron_density,
    toroidal_flux_expansion,
    average_ion_mass,
    kappa_e0,
    SOL_momentum_loss_function,
    sheath_heat_transmission_factor,
):
    return formulas.scrape_off_layer.two_point_model_fixed_tet(
        target_electron_temp=target_electron_temp,
        q_parallel=q_parallel,
        parallel_connection_length=parallel_connection_length,
        separatrix_electron_density=separatrix_electron_density,
        toroidal_flux_expansion=toroidal_flux_expansion,
        average_ion_mass=average_ion_mass,
        kappa_e0=kappa_e0,
        SOL_momentum_loss_function=SOL_momentum_loss_function,
        sheath_heat_transmission_factor=sheath_heat_transmission_factor,
    )


@pytest.fixture()
def target_neutral_pressure(
    average_ion_mass,
    kappa_e0,
    kappa_ez,
    parallel_connection_length,
    target_angle_of_incidence,
    lambda_q,
    target_gaussian_spreading,
    sheath_heat_transmission_factor,
    neutral_flux_density_factor,
    SOL_power_loss_fraction,
    SOL_momentum_loss_function,
    separatrix_electron_density,
    target_electron_temp,
    q_parallel,
):
    return formulas.scrape_off_layer.calc_neutral_pressure_kallenbach(
        average_ion_mass=average_ion_mass,
        kappa_e0=kappa_e0,
        kappa_ez=kappa_ez,
        parallel_connection_length=parallel_connection_length,
        target_angle_of_incidence=target_angle_of_incidence,
        lambda_q=lambda_q,
        target_gaussian_spreading=target_gaussian_spreading,
        sheath_heat_transmission_factor=sheath_heat_transmission_factor,
        neutral_flux_density_factor=neutral_flux_density_factor,
        SOL_power_loss_fraction=SOL_power_loss_fraction,
        SOL_momentum_loss_function=SOL_momentum_loss_function,
        separatrix_electron_density=separatrix_electron_density,
        target_electron_temp=target_electron_temp,
        q_parallel=q_parallel,
    )


def test_calc_reattachment_time_henderson(
    target_neutral_pressure,
    target_electron_density,
    major_radius,
    parallel_connection_length,
    separatrix_power_transient,
):
    reattachment_time = formulas.scrape_off_layer.calc_reattachment_time_henderson(
        target_neutral_pressure=target_neutral_pressure,
        target_electron_density=target_electron_density,
        major_radius=major_radius,
        parallel_connection_length=parallel_connection_length,
        separatrix_power_transient=separatrix_power_transient,
    )

    reattachment_time = reattachment_time.item().to_base_units().magnitude

    assert np.isclose(reattachment_time, 0.81782812)

    return
