import numpy as np
import pytest
import xarray as xr

from cfspopcon.named_options import MomentumLossFunction
from cfspopcon.unit_handling import Quantity, ureg, get_units, convert_units
from cfspopcon.unit_handling import dimensionless_magnitude as dmag
from cfspopcon.unit_handling import magnitude_in_units as umag
from cfspopcon import formulas


@pytest.fixture()
def magnetic_field_on_axis():
    """Toroidal field on-axis in Tesla."""
    return Quantity(12.2, ureg.T)


@pytest.fixture()
def major_radius():
    """Major radius in metres."""
    return Quantity(1.85, ureg.m)


@pytest.fixture()
def minor_radius():
    """Minor radius in metres."""
    return Quantity(0.55, ureg.m)


@pytest.fixture()
def average_ion_mass():
    """Ion mass in amu."""
    return Quantity(2.515, ureg.amu)


@pytest.fixture()
def plasma_current():
    """Plasma current in A."""
    return Quantity(8.7e6, ureg.A)


@pytest.fixture()
def elongation_psi95():
    """Elongation (kappa) at the psiN = 0.95 flux surface."""
    return 1.68


@pytest.fixture()
def triangularity_psi95():
    """Triangularity (delta) at the psiN = 0.95 flux surface."""
    return 0.3


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
    return Quantity(2400.0, ureg.watt / ureg.electron_volt**3.5 / ureg.meter)


@pytest.fixture()
def SOL_momentum_loss_function():
    return MomentumLossFunction.KotovReiter


@pytest.fixture()
def target_angle_of_incidence():
    return Quantity(2.0, ureg.degree)


@pytest.fixture()
def kappa_ez():
    return 4.0


@pytest.fixture()
def sheath_heat_transmission_factor():
    return 7.5


@pytest.fixture()
def separatrix_power_transient():
    return Quantity(5.0, ureg.MW)


@pytest.fixture()
def SOL_power_loss_fraction():
    return 0.9597046482


@pytest.fixture()
def ionization_volume_density_factor():
    return 1.0


@pytest.fixture()
def ratio_of_divertor_to_duct_pressure():
    return 1.0


@pytest.fixture()
def ratio_of_molecular_to_ion_mass():
    return 2.0


@pytest.fixture()
def wall_temperature():
    return Quantity(300.0, ureg.K)


@pytest.fixture()
def areal_elongation():
    return 1.75


@pytest.fixture()
def inverse_aspect_ratio():
    return 0.3081


@pytest.fixture()
def plasma_volume(
    major_radius,
    areal_elongation,
    inverse_aspect_ratio,
):
    v = formulas.geometry.calc_plasma_volume(
        major_radius=major_radius,
        areal_elongation=areal_elongation,
        inverse_aspect_ratio=inverse_aspect_ratio,
    )
    return v


@pytest.fixture()
def cylindrical_safety_factor(
    magnetic_field_on_axis,
    major_radius,
    minor_radius,
    plasma_current,
    elongation_psi95,
    triangularity_psi95,
):
    q_cyl = formulas.plasma_current.safety_factor.calc_cylindrical_edge_safety_factor(
        major_radius=major_radius,
        minor_radius=minor_radius,
        elongation_psi95=elongation_psi95,
        triangularity_psi95=triangularity_psi95,
        magnetic_field_on_axis=magnetic_field_on_axis,
        plasma_current=plasma_current,
    )
    return convert_units(q_cyl, ureg.dimensionless)


@pytest.fixture()
def q_parallel(
    power_crossing_separatrix,
    fraction_of_P_SOL_to_divertor,
    major_radius,
    minor_radius,
    lambda_q,
    fieldline_pitch_at_omp,
):
    q_par = formulas.scrape_off_layer.heat_flux_density.calc_parallel_heat_flux_density(
        power_crossing_separatrix=power_crossing_separatrix,
        fraction_of_P_SOL_to_divertor=fraction_of_P_SOL_to_divertor,
        major_radius=major_radius,
        minor_radius=minor_radius,
        lambda_q=lambda_q,
        fieldline_pitch_at_omp=fieldline_pitch_at_omp,
    )
    return convert_units(q_par, ureg.W / ureg.m**2)


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
    Te_tar = formulas.scrape_off_layer.two_point_model_fixed_tet(
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
    return convert_units(Te_tar, ureg.eV)


@pytest.fixture()
def neutral_flux_density_factor(
    average_ion_mass,
    ratio_of_molecular_to_ion_mass,
    wall_temperature,
):
    factor = formulas.scrape_off_layer.calc_neutral_flux_density_factor(
        average_ion_mass=average_ion_mass,
        ratio_of_molecular_to_ion_mass=ratio_of_molecular_to_ion_mass,
        wall_temperature=wall_temperature,
    )
    return factor


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
    ratio_of_divertor_to_duct_pressure,
):
    p_div, p_duct = formulas.scrape_off_layer.calc_neutral_pressure_kallenbach(
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
        ratio_of_divertor_to_duct_pressure=ratio_of_divertor_to_duct_pressure,
    )

    return convert_units(p_div, ureg.Pa)


@pytest.fixture()
def ionization_volume(
    plasma_volume,
):
    ionization_volume = formulas.scrape_off_layer.calc_ionization_volume_from_AUG(plasma_volume=plasma_volume)
    return ionization_volume


def test_calc_reattachment_time_henderson(
    target_neutral_pressure,
    target_electron_density,
    parallel_connection_length,
    separatrix_power_transient,
    ionization_volume_density_factor,
    ionization_volume,
):
    reattachment_time = formulas.scrape_off_layer.calc_reattachment_time_henderson(
        target_neutral_pressure=target_neutral_pressure,
        target_electron_density=target_electron_density,
        parallel_connection_length=parallel_connection_length,
        separatrix_power_transient=separatrix_power_transient,
        ionization_volume_density_factor=ionization_volume_density_factor,
        ionization_volume=ionization_volume,
    )
    assert np.isclose(umag(reattachment_time, ureg.s), 1.23233809)

    return
