"""Calculate the energy confinement from scalings."""

import numpy as np

from ...algorithm_class import Algorithm, CompositeAlgorithm
from ...unit_handling import Unitfull, ureg, wraps_ufunc
from .read_energy_confinement_scalings import _get_confinement_scaling


@Algorithm.register_algorithm(return_keys=["energy_confinement_time"])
@wraps_ufunc(
    return_units=dict(tau_e=ureg.s),
    input_units=dict(
        confinement_time_scalar=ureg.dimensionless,
        plasma_current=ureg.MA,
        magnetic_field_on_axis=ureg.T,
        average_electron_density=ureg.n19,
        major_radius=ureg.m,
        areal_elongation=ureg.dimensionless,
        separatrix_elongation=ureg.dimensionless,
        inverse_aspect_ratio=ureg.dimensionless,
        average_ion_mass=ureg.amu,
        triangularity_psi95=ureg.dimensionless,
        separatrix_triangularity=ureg.dimensionless,
        P_in=ureg.MW,
        q_star=ureg.dimensionless,
        energy_confinement_scaling=None,
    ),
)
def calc_energy_confinement_time_from_scaling(
    confinement_time_scalar: float,
    plasma_current: float,
    magnetic_field_on_axis: float,
    average_electron_density: float,
    major_radius: float,
    areal_elongation: float,
    separatrix_elongation: float,
    inverse_aspect_ratio: float,
    average_ion_mass: float,
    triangularity_psi95: float,
    separatrix_triangularity: float,
    P_in: float,
    q_star: float,
    energy_confinement_scaling: str,
) -> float:
    """Calculate the energy confinement time from a scaling, given a known input power.

    Args:
        confinement_time_scalar: [~] confinement scaling factor
        plasma_current: [MA] :term:`glossary link<plasma_current>`
        magnetic_field_on_axis: [T] :term:`glossary link<magnetic_field_on_axis>`
        average_electron_density: [1e19 m^-3] :term:`glossary link<average_electron_density>`
        major_radius: [m] :term:`glossary link<major_radius>`
        areal_elongation: [~] :term:`glossary link<areal_elongation>`
        separatrix_elongation: [~] :term:`glossary link<separatrix_elongation>`
        inverse_aspect_ratio: [~] :term:`glossary link<inverse_aspect_ratio>`
        average_ion_mass: [amu] :term:`glossary link<average_ion_mass>`
        triangularity_psi95: [~] :term:`glossary link<triangularity_psi95>`
        separatrix_triangularity: [~] :term:`glossary link<separatrix_triangularity>`
        P_in: [MJ] :term:`glossary link<P_in>`
        q_star: [~] :term:`glossary link<q_star>`
        energy_confinement_scaling: [] :term:`glossary link<energy_confinement_scaling>`

    Returns:
        :term:`energy_confinement_time` [s]
    """
    scaling = _get_confinement_scaling(energy_confinement_scaling)

    # Avoid passing zero power into a negative input-power exponent.
    clipped_input_power = np.maximum(P_in, 1e-3)

    tau_e = (
        confinement_time_scalar
        * scaling.constant
        * plasma_current**scaling.plasma_current_alpha
        * magnetic_field_on_axis**scaling.field_on_axis_alpha
        * average_electron_density**scaling.average_density_alpha
        * major_radius**scaling.major_radius_alpha
        * areal_elongation**scaling.areal_elongation_alpha
        * separatrix_elongation**scaling.separatrix_elongation_alpha
        * inverse_aspect_ratio**scaling.inverse_aspect_ratio_alpha
        * average_ion_mass**scaling.mass_ratio_alpha
        * (1.0 + np.mean([triangularity_psi95, separatrix_triangularity])) ** scaling.triangularity_alpha
        * q_star**scaling.qstar_alpha
        * clipped_input_power**scaling.input_power_alpha
    )

    return tau_e


@Algorithm.register_algorithm(return_keys=["energy_confinement_time"])
def calc_energy_confinement_time_from_stored_energy_and_input_power(plasma_stored_energy: Unitfull, P_in: Unitfull) -> Unitfull:
    """Calculate the energy confinement time according to the definition, given a known input power and stored energy.

    Args:
        plasma_stored_energy: [MJ] :term:`glossary link<plasma_stored_energy>`
        P_in: [MW] :term:`glossary link<P_in>`

    Returns:
        :term:`energy_confinement_time` [s]
    """
    energy_confinement_time = plasma_stored_energy / np.maximum(P_in, 1e-3 * ureg.MW)
    return energy_confinement_time


@Algorithm.register_algorithm(return_keys=["H98y2"])
def calc_H98y2(
    energy_confinement_time: Unitfull,
    plasma_current: Unitfull,
    magnetic_field_on_axis: Unitfull,
    average_electron_density: Unitfull,
    major_radius: Unitfull,
    areal_elongation: Unitfull,
    separatrix_elongation: Unitfull,
    inverse_aspect_ratio: Unitfull,
    average_ion_mass: Unitfull,
    triangularity_psi95: Unitfull,
    separatrix_triangularity: Unitfull,
    P_in: Unitfull,
    q_star: Unitfull,
) -> Unitfull:
    """Calculate the ratio of the energy confinement to the energy confinement according to the ITER98y2 scaling.

    Args:
        energy_confinement_time: [s] :term:`glossary link<energy_confinement_time>`
        plasma_current: [MA] :term:`glossary link<plasma_current>`
        magnetic_field_on_axis: [T] :term:`glossary link<magnetic_field_on_axis>`
        average_electron_density: [1e19 m^-3] :term:`glossary link<average_electron_density>`
        major_radius: [m] :term:`glossary link<major_radius>`
        areal_elongation: [~] :term:`glossary link<areal_elongation>`
        separatrix_elongation: [~] :term:`glossary link<separatrix_elongation>`
        inverse_aspect_ratio: [~] :term:`glossary link<inverse_aspect_ratio>`
        average_ion_mass: [amu] :term:`glossary link<average_ion_mass>`
        triangularity_psi95: [~] :term:`glossary link<triangularity_psi95>`
        separatrix_triangularity: [~] :term:`glossary link<separatrix_triangularity>`
        P_in: [MJ] :term:`glossary link<P_in>`
        q_star: [~] :term:`glossary link<q_star>`

    Returns:
        :term:`H98y2` [~]
    """
    tau_e_98y2 = calc_energy_confinement_time_from_scaling(
        confinement_time_scalar=1.0,
        plasma_current=plasma_current,
        magnetic_field_on_axis=magnetic_field_on_axis,
        average_electron_density=average_electron_density,
        major_radius=major_radius,
        areal_elongation=areal_elongation,
        separatrix_elongation=separatrix_elongation,
        inverse_aspect_ratio=inverse_aspect_ratio,
        average_ion_mass=average_ion_mass,
        triangularity_psi95=triangularity_psi95,
        separatrix_triangularity=separatrix_triangularity,
        P_in=P_in,
        q_star=q_star,
        energy_confinement_scaling="ITER98y2",
    )
    return energy_confinement_time / tau_e_98y2


calc_power_balance_from_input_P_aux = CompositeAlgorithm.from_list(
    ["calc_input_power_for_fixed_auxiliary_power", "calc_energy_confinement_time_from_stored_energy_and_input_power", "calc_H98y2"]
)
