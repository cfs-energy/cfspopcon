"""Define default units for writing to/from disk."""
from collections.abc import Iterable
from numbers import Number
from typing import Any, Union, overload

import numpy as np
import xarray as xr

from .setup_unit_handling import DimensionalityError, Quantity, convert_units, magnitude

DEFAULT_UNITS = dict(
    areal_elongation="",
    average_electron_density="n19",
    average_electron_temp="keV",
    average_ion_density="n19",
    average_ion_temp="keV",
    average_total_pressure="Pa",
    B_pol_out_mid="T",
    B_t_out_mid="T",
    beta_poloidal="",
    beta_toroidal="",
    beta="",
    bootstrap_fraction="",
    confinement_threshold_scalar="",
    confinement_time_scalar="",
    core_radiated_power_fraction="",
    core_radiator_charge_state="",
    core_radiator_concentration="",
    core_radiator=None,
    current_relaxation_time="s",
    dilution_change_from_core_rad="",
    dilution="",
    effective_collisionality="",
    electron_density_peaking_offset="",
    electron_density_peaking="",
    electron_density_profile="n19",
    electron_temp_profile="keV",
    elongation_ratio_sep_to_areal="",
    energy_confinement_scaling=None,
    energy_confinement_time="s",
    f_shaping="",
    fieldline_pitch_at_omp="",
    fraction_of_external_power_coupled="",
    fraction_of_P_SOL_to_divertor="",
    fuel_average_mass_number="amu",
    fuel_ion_density_profile="n19",
    fusion_reaction=None,
    fusion_triple_product="n20 * keV * s",
    greenwald_fraction="",
    heavier_fuel_species_fraction="",
    impurities="",
    impurity_charge_state="",
    input_SOL_power_loss_fraction="",
    input_target_electron_temp="eV",
    input_target_q_parallel="GW / m**2",
    inverse_aspect_ratio="",
    ion_density_peaking_offset="",
    ion_density_peaking="",
    ion_temp_profile="keV",
    ion_to_electron_temp_ratio="",
    kappa_e0="W / (eV**3.5 m)",
    lambda_q_factor="",
    lambda_q_scaling=None,
    lambda_q="mm",
    loop_voltage="V",
    magnetic_field_on_axis="T",
    major_radius="m",
    minimum_core_radiated_fraction="",
    minor_radius="m",
    neoclassical_loop_resistivity="m * ohm",
    nesep_over_nebar="",
    neutron_power_flux_to_walls="MW / m**2",
    neutron_rate="s**-1",
    normalized_beta="percent * m * T / MA",
    normalized_inverse_temp_scale_length="",
    nu_star="",
    P_alpha="MW",
    P_auxillary="MW",
    P_external="MW",
    P_fusion="MW",
    P_in="MW",
    P_launched="MW",
    P_LH_thresh="MW",
    P_neutron="MW",
    P_ohmic="MW",
    P_radiated_by_core_radiator="MW",
    P_radiation="MW",
    P_sol="MW",
    parallel_connection_length="m",
    PB_over_R="MW * T / m",
    PBpRnSq="MW * T / m * n20**-2",
    peak_electron_density="n19",
    peak_electron_temp="keV",
    peak_fuel_ion_density="n19",
    peak_ion_temp="keV",
    peak_pressure="Pa",
    plasma_current="A",
    plasma_stored_energy="MJ",
    plasma_volume="m**3",
    product_of_magnetic_field_and_radius="m * T",
    profile_form=None,
    q_parallel="GW / m**2",
    q_perp="MW / m**2",
    q_star="",
    Q="",
    radiated_power_method=None,
    radiated_power_scalar="",
    ratio_of_P_SOL_to_P_LH="",
    rho_star="",
    rho="",
    separatrix_elongation="",
    separatrix_triangularity="",
    SOC_LOC_ratio="",
    SOL_momentum_loss_function=None,
    SOL_power_loss_fraction="",
    spitzer_resistivity="m * ohm",
    summed_impurity_density="n19",
    surface_area="m**2",
    target_electron_density="n19",
    target_electron_flux="m**-2 s**-1",
    target_electron_temp="eV",
    target_q_parallel="GW / m**2",
    tau_e_scaling_uses_P_in=None,
    temperature_peaking="",
    toroidal_flux_expansion="",
    trapped_particle_fraction="",
    triangularity_psi95="",
    triangularity_ratio_sep_to_psi95="",
    two_point_model_method=None,
    upstream_electron_temp="eV",
    vertical_minor_radius="m",
    z_effective="",
    zeff_change_from_core_rad="",
)


def default_unit(var: str) -> Union[str, None]:
    """Return cfspopcon's default unit for a given quantity.

    Args:
        var: Quantity name

    Returns: Unit
    """
    try:
        return DEFAULT_UNITS[var]
    except KeyError:
        raise KeyError(
            f"No default unit defined for {var}. Please check configured default units in the unit_handling submodule."
        ) from None


def magnitude_in_default_units(value: Union[Quantity, xr.DataArray], key: str) -> Union[float, list[float], Any]:
    """Convert values to default units and then return the magnitude.

    Args:
        value: input value to convert to a float
        key: name of field for looking up in DEFAULT_UNITS dictionary

    Returns:
        magnitude of value in default units and as basic type
    """
    try:
        # unit conversion step
        unit = default_unit(key)
        if unit is None:
            return value

        mag = magnitude(convert_units(value, unit))

    except DimensionalityError as e:
        print(f"Unit conversion failed for {key}. Could not convert '{value}' to '{DEFAULT_UNITS[key]}'")
        raise e

    # single value arrays -> float
    # np,xr array -> list
    if isinstance(mag, (np.ndarray, xr.DataArray)):
        if mag.size == 1:
            return float(mag)
        else:
            return [float(v) for v in mag]
    else:
        return float(mag)


@overload
def set_default_units(value: Number, key: str) -> Quantity:
    ...


@overload
def set_default_units(value: xr.DataArray, key: str) -> xr.DataArray:
    ...


@overload
def set_default_units(value: Any, key: str) -> Any:
    ...


def set_default_units(value: Any, key: str) -> Any:
    """Return value as a quantity with default units.

    Args:
        value: magnitude of input value to convert to a Quantity
        key: name of field for looking up in DEFAULT_UNITS dictionary

    Returns:
        magnitude of value in default units
    """

    def _is_number_not_bool(val: Any) -> bool:
        return isinstance(val, Number) and not isinstance(val, bool)

    def _is_iterable_of_number_not_bool(val: Any) -> bool:
        if not isinstance(val, Iterable):
            return False

        if isinstance(val, (np.ndarray, xr.DataArray)) and val.ndim == 0:
            return _is_number_not_bool(val.item())

        return all(_is_number_not_bool(v) for v in value)

    # None is used to ignore class types
    if DEFAULT_UNITS[key] is None:
        if _is_number_not_bool(value) or _is_iterable_of_number_not_bool(value):
            raise RuntimeError(
                f"set_default_units for key {key} and value {value} of type {type(value)}: numeric types should carry units!"
            )
        return value
    elif isinstance(value, xr.DataArray):
        return value.pint.quantify(DEFAULT_UNITS[key])
    else:
        return Quantity(value, DEFAULT_UNITS[key])


@overload
def convert_to_default_units(value: float, key: str) -> float:
    ...


@overload
def convert_to_default_units(value: xr.DataArray, key: str) -> xr.DataArray:
    ...


@overload
def convert_to_default_units(value: Quantity, key: str) -> Quantity:
    ...


def convert_to_default_units(value: Union[float, Quantity, xr.DataArray], key: str) -> Union[float, Quantity, xr.DataArray]:
    """Convert an array or scalar to default units."""
    unit = DEFAULT_UNITS[key]
    if unit is None:
        return value
    elif isinstance(value, (xr.DataArray, Quantity)):
        return convert_units(value, unit)
    else:
        raise NotImplementedError(f"No implementation for 'convert_to_default_units' with an array of type {type(value)} ({value})")
