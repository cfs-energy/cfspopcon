"""Resistive flux consumption, inductive flux consumption (internally and externally on the plasma surface) during purely ohmic ramp-up.

TODO: Isaac please check this.
"""

import numpy as np
import xarray as xr

from ....algorithm_class import Algorithm
from ....named_options import SurfaceInductanceCoeffs, VertMagneticFieldEq
from ....unit_handling import Unitfull, ureg
from ....unit_handling import dimensionless_magnitude as dmag
from .inductance_analytical_functions import (
    calc_fa,
    calc_fb,
    calc_fc,
    calc_fd,
    calc_fg,
    calc_fh,
)


def set_surface_inductance_coeffs(
    surface_inductance_coefficients: SurfaceInductanceCoeffs,
) -> dict[str, np.ndarray]:
    """Choose which coefficients you want to use for the external flux calculation.

    1. Barr's Coefficients cite:`Barr_2018`.
    2. Hirshman's Coefficients cite:'hirshman'.

    """
    if isinstance(surface_inductance_coefficients, xr.DataArray):
        surface_inductance_coefficients = surface_inductance_coefficients.item()
    if isinstance(surface_inductance_coefficients, str):
        surface_inductance_coefficients = SurfaceInductanceCoeffs[
            surface_inductance_coefficients
        ]

    if surface_inductance_coefficients == SurfaceInductanceCoeffs.Barr:
        return dict(
            a=np.array([1.438, 2.139, 9.387, -1.939]),
            b=np.array([0.149, 1.068, -6.216, 4.126]),
            c=np.array([-0.293, -0.349, 0.098]),
            d=np.array([0.003, 0.334, -2.018]),
            e=np.array([0.080, -0.260, -0.267, 1.135]),
        )
    elif surface_inductance_coefficients == SurfaceInductanceCoeffs.Hirshman:
        return dict(
            a=np.array([1.81, 2.05, 9.25, -1.21]),
            b=np.array([0.73, 2, -6.00, 3.70]),
            c=np.array([0.98, 0.49, 1.47]),
            d=np.array([0.25, 0.84, -1.44]),
            e=np.array([0, 0, 0, 0]),  # not available
        )
    else:
        raise NotImplementedError(
            f"Unrecognised SurfaceInductanceCoeffs option {surface_inductance_coefficients.name}"
        )


@Algorithm.register_algorithm(return_keys=["internal_inductivity"])
def calc_internal_inductivity(
    cylindrical_safety_factor: Unitfull,
    safety_factor_on_axis: Unitfull = 1.0,
) -> Unitfull:
    """Calculate the normalized internal inductance for an assumed circular plasma cross-section.

    Tokamaks (pg.120): Physics :cite:`wesson_tokamaks_2011`

    TODO: Isaac please check if the implementation for cylindrical safety factor is consistent
    TODO: previous: cylindrical_safety_factor = 2 * np.pi * minor_radius**2 * magnetic_field_on_axis / (constants.mu_0 * major_radius * plasma_current)

    Args:
        cylindrical_safety_factor: [~] :term:`glossary link<cylindrical_safety_factor>`
        safety_factor_on_axis: [~] :term:`glossary link<safety_factor_on_axis>`

    Returns:
        [~] :term:`internal_inductivity`
    """
    return np.log(
        1.65 + 0.89 * ((cylindrical_safety_factor / safety_factor_on_axis) - 1.0)
    )


@Algorithm.register_algorithm(return_keys=["internal_inductance"])
def calc_internal_inductance_for_cylindrical(
    major_radius: Unitfull, internal_inductivity: Unitfull
) -> Unitfull:
    """Calculate the internal inductance of the plasma (assuming a circular cross-section).

    TODO: what is the difference between inductivity and inductance?

    A power-balance model for local helicity injection startup in a spherical tokamak :cite:`Barr_2018`

    Returns:
        [henry] :term:`internal_inductance`
    """
    return ureg.mu_0 * major_radius * internal_inductivity / 2.0


@Algorithm.register_algorithm(return_keys=["internal_inductance"])
def calc_internal_inductance_for_noncylindrical(
    plasma_volume: Unitfull,
    poloidal_circumference: Unitfull,
    internal_inductivity: Unitfull,
) -> Unitfull:
    """Calculate the internal inductance of the plasma.

    TODO: what is the difference between inductivity and inductance?

    A power-balance model for local helicity injection startup in a spherical tokamak :cite:`Barr_2018`

    Returns:
        [henry] :term:`internal_inductance`
    """
    return (
        ureg.mu_0 * internal_inductivity * plasma_volume / (poloidal_circumference**2)
    )


@Algorithm.register_algorithm(return_keys=["external_inductance"])
def calc_external_inductance(
    inverse_aspect_ratio: Unitfull,
    areal_elongation: Unitfull,
    beta_poloidal: Unitfull,
    major_radius: Unitfull,
    internal_inductivity: Unitfull,
    surface_inductance_coefficients: SurfaceInductanceCoeffs,
) -> Unitfull:
    """Calculate the external self-inductance of the plasma for which the current-induced surface flux of the plasma is generated from eq. 13 in :cite:`Barr_2018`.

    A power-balance model for local helicity injection startup in a spherical tokamak :cite:`Barr_2018`

    Args:
        inverse_aspect_ratio: [~] :term:`glossary link<inverse_aspect_ratio>`
        areal_elongation: [~] :term:`glossary link<areal_elongation>`
        beta_poloidal: [~] :term:`glossary link<beta_poloidal>`
        major_radius: [m] :term:`glossary link<major_radius>`
        internal_inductivity: [~] :term:`glossary link<internal_inductivity>`
        surface_inductance_coefficients: [~] :term:`glossary link<surface_inductance_coefficients>`

    Returns:
        [henry] :term:`external_inductance`
    """
    coeffs = set_surface_inductance_coeffs(surface_inductance_coefficients)

    fa = calc_fa(
        inverse_aspect_ratio=inverse_aspect_ratio,
        beta_poloidal=beta_poloidal,
        internal_inductivity=internal_inductivity,
        coeffs=coeffs,
    )
    fb = calc_fb(inverse_aspect_ratio=inverse_aspect_ratio, coeffs=coeffs)

    return (
        ureg.mu_0
        * major_radius
        * fa
        * (1 - inverse_aspect_ratio)
        / ((1 - inverse_aspect_ratio) + areal_elongation * fb)
    )


@Algorithm.register_algorithm(return_keys=["vertical_field_mutual_inductance"])
def calc_vertical_field_mutual_inductance(
    inverse_aspect_ratio: Unitfull,
    areal_elongation: Unitfull,
    surface_inductance_coefficients: SurfaceInductanceCoeffs,
) -> Unitfull:
    """Calculate the mutual inductance linking the surface to the vertical field from eq. 15 in :cite:`Barr_2018`.

    A power-balance model for local helicity injection startup in a spherical tokamak :cite:`Barr_2018`

    Args:
        inverse_aspect_ratio: [~] :term:`glossary link<inverse_aspect_ratio>`
        areal_elongation: [~] :term:`glossary link<areal_elongation>`
        surface_inductance_coefficients: [~] :term:`glossary link<surface_inductance_coefficients>`

    Returns:
        [~] :term:`vertical_field_mutual_inductance`
    """
    coeffs = set_surface_inductance_coeffs(surface_inductance_coefficients)

    fc = calc_fc(inverse_aspect_ratio=inverse_aspect_ratio, coeffs=coeffs)
    fd = calc_fd(inverse_aspect_ratio=inverse_aspect_ratio, coeffs=coeffs)

    return (1 - inverse_aspect_ratio) ** 2 / (
        (1 - inverse_aspect_ratio) ** 2 * fc + fd * np.sqrt(areal_elongation)
    )


@Algorithm.register_algorithm(return_keys=["invmu_0_dLedR"])
def calc_invmu_0_dLedR(
    inverse_aspect_ratio: Unitfull,
    areal_elongation: Unitfull,
    beta_poloidal: Unitfull,
    internal_inductivity: Unitfull,
    external_inductance: Unitfull,
    major_radius: Unitfull,
    surface_inductance_coefficients: SurfaceInductanceCoeffs,
) -> Unitfull:
    """Calculate eq. 21 on page 6 in :cite:`Barr_2018`.

    A power-balance model for local helicity injection startup in a spherical tokamak :cite:`Barr_2018`

    Args:
        inverse_aspect_ratio: [~] :term:`glossary link<inverse_aspect_ratio>`
        areal_elongation: [~] :term:`glossary link<areal_elongation>`
        beta_poloidal: [~] :term:`glossary link<beta_poloidal>`
        internal_inductivity: [~] :term:`glossary link<internal_inductivity>`
        external_inductance: [henry] :term:`glossary link<external_inductance>`
        major_radius: [m] :term:`glossary link<major_radius>`
        surface_inductance_coefficients: [~] :term:`glossary link<surface_inductance_coefficients>`

    Returns:
        [~] :term:`invmu_0_dLedR`
    """
    coeffs = set_surface_inductance_coeffs(surface_inductance_coefficients)

    fa = calc_fa(
        inverse_aspect_ratio=inverse_aspect_ratio,
        beta_poloidal=beta_poloidal,
        internal_inductivity=internal_inductivity,
        coeffs=coeffs,
    )
    fb = calc_fb(inverse_aspect_ratio=inverse_aspect_ratio, coeffs=coeffs)
    fg = calc_fg(
        inverse_aspect_ratio=inverse_aspect_ratio,
        beta_poloidal=beta_poloidal,
        internal_inductivity=internal_inductivity,
        coeffs=coeffs,
    )
    fh = calc_fh(
        inverse_aspect_ratio=inverse_aspect_ratio,
        areal_elongation=areal_elongation,
        coeffs=coeffs,
    )

    invmu_0_dLedR = (1 / ureg.mu_0) * (
        ureg.mu_0
        * inverse_aspect_ratio
        * (1 - inverse_aspect_ratio)
        * fa
        * fh
        / (((1 - inverse_aspect_ratio) + areal_elongation * fb) ** 2)
        - ureg.mu_0
        * inverse_aspect_ratio
        * (1 - inverse_aspect_ratio)
        * fg
        / ((1 - inverse_aspect_ratio) + areal_elongation * fb)
        + (inverse_aspect_ratio)
        * ureg.mu_0
        * fa
        / ((1 - inverse_aspect_ratio) + areal_elongation * fb)
        + external_inductance / major_radius
    )

    return invmu_0_dLedR


@Algorithm.register_algorithm(return_keys=["vertical_magnetic_field"])
def calc_vertical_magnetic_field(
    inverse_aspect_ratio: Unitfull,
    areal_elongation: Unitfull,
    beta_poloidal: Unitfull,
    internal_inductivity: Unitfull,
    external_inductance: Unitfull,
    major_radius: Unitfull,
    plasma_current: Unitfull,
    invmu_0_dLedR: Unitfull = 0,
    vertical_magnetic_field_equation: VertMagneticFieldEq = VertMagneticFieldEq.Barr,
    surface_inductance_coefficients: SurfaceInductanceCoeffs = SurfaceInductanceCoeffs.Hirshman,
) -> Unitfull:
    """Calculate the mutual inductance linking the surface to the vertical field from eq. 16 in :cite:`Barr_2018`.

    A power-balance model for local helicity injection startup in a spherical tokamak :cite:`Barr_2018`

    Args:
        inverse_aspect_ratio: [~] :term:`glossary link<inverse_aspect_ratio>`
        areal_elongation: [~] :term:`glossary link<areal_elongation>`
        beta_poloidal: [~] :term:`glossary link<beta_poloidal>`
        internal_inductivity: [~] :term:`glossary link<internal_inductivity>`
        external_inductance: [~] :term:`glossary link<external_inductance>`
        major_radius: [m] :term:`glossary link<major_radius>`
        plasma_current: [A] :term:`glossary link<plasma_current>`
        invmu_0_dLedR: [~] :term:`glossary link<invmu_0_dLedR>`
        vertical_magnetic_field_equation: [~] :term:`glossary link<vertical_magnetic_field_equation>`
        surface_inductance_coefficients: [~] :term:`glossary link<surface_inductance_coefficients>`

    Returns:
        [T] :term:`vertical_magnetic_field`
    """
    if isinstance(vertical_magnetic_field_equation, xr.DataArray):
        vertical_magnetic_field_equation = vertical_magnetic_field_equation.item()
    if isinstance(vertical_magnetic_field_equation, str):
        vertical_magnetic_field_equation = VertMagneticFieldEq[
            vertical_magnetic_field_equation
        ]

    if vertical_magnetic_field_equation == VertMagneticFieldEq.Barr:
        assert np.all(
            np.abs(dmag(invmu_0_dLedR)) > 0
        ), "Cannot compute Barr vertical magnetic field with invmu_0_dLedR = 0."

        vertical_magnetic_field = (
            ureg.mu_0
            * plasma_current
            * (1 / (4 * np.pi * major_radius))
            * (invmu_0_dLedR + (beta_poloidal + (internal_inductivity / 2)) - (1 / 2))
        )
    elif (
        vertical_magnetic_field_equation
        == VertMagneticFieldEq.MagneticFusionEnergyFormulary
    ):
        vertical_magnetic_field = (
            ureg.mu_0
            * plasma_current
            * (1 / (4 * np.pi * major_radius))
            * (
                np.log(8 / inverse_aspect_ratio)
                + beta_poloidal
                + (internal_inductivity / 2)
                - 1.5
            )
        )
    elif vertical_magnetic_field_equation == VertMagneticFieldEq.Mit_and_Taka_Eq13:
        vertical_magnetic_field = (
            ureg.mu_0
            * plasma_current
            * (1 / (4 * np.pi * major_radius))
            * (
                np.log(
                    8
                    / (
                        inverse_aspect_ratio
                        * (np.sqrt((1 + areal_elongation**2) / 2))
                    )
                )
                + beta_poloidal
                + (internal_inductivity / 2)
                - 1.5
            )
        )
    elif vertical_magnetic_field_equation == VertMagneticFieldEq.Jean:
        vertical_magnetic_field = (
            ureg.mu_0
            * plasma_current
            * (1 / (4 * np.pi * major_radius))
            * (
                np.log(8 / (inverse_aspect_ratio * (np.sqrt(areal_elongation))))
                + beta_poloidal
                + (internal_inductivity / 2)
                - 1.5
            )
        )
    else:
        raise NotImplementedError(
            f"Unrecognised VertMagneticField option {vertical_magnetic_field_equation.name}"
        )

    return vertical_magnetic_field
