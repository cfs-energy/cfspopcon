"""Calculate the vertical magnetic field, as well as the plasma surface's mutual inductance with the vertical field, internal inductivity, external inductance and internal inductance."""
from .. import formulas, named_options
from ..unit_handling import Unitfull, convert_to_default_units
from .algorithm_class import Algorithm

RETURN_KEYS = [
    "internal_inductivity",
    "internal_inductance",
    "external_inductance",
    "vertical_field_mutual_inductance",
    "vertical_magnetic_field",
]


def run_calc_inductances(
    major_radius: Unitfull,
    plasma_volume: Unitfull,
    poloidal_circumference: Unitfull,
    internal_inductance_geometry: named_options.InternalInductanceGeometry,
    plasma_current: Unitfull,
    magnetic_field_on_axis: Unitfull,
    minor_radius: Unitfull,
    safety_factor_on_axis: Unitfull,
    inverse_aspect_ratio: Unitfull,
    areal_elongation: Unitfull,
    beta_poloidal: Unitfull,
    vertical_magnetic_field_equation: named_options.VertMagneticFieldEq,
    surface_inductance_coefficients: named_options.SurfaceInductanceCoeffs,
    internal_inductivity: Unitfull = None,
) -> dict[str, Unitfull]:
    """Calculate the vertical magnetic field, as well as the plasma surface's mutual inductance with the vertical field, internal inductivity, external inductance and internal inductance.

    Args:
        major_radius: :term:`glossary link<major_radius>`
        plasma_volume: [m**3] :term:`glossary<plasma_volume>`
        poloidal_circumference: [m] :term:`glossary<poloidal_circumference>`
        internal_inductance_geometry: [~] :term:`glossary<internal_inductance_geometry>`
        plasma_current: :term:`glossary link<plasma_current>`
        magnetic_field_on_axis: [T] :term:`glossary<magnetic_field_on_axis>`
        minor_radius: :term:`glossary link<minor_radius>`
        safety_factor_on_axis: [~] :term:`glossary<safety_factor_on_axis>`
        inverse_aspect_ratio: [~] :term:`glossary link<inverse_aspect_ratio>`
        areal_elongation: [~] :term:`glossary<areal_elongation>`
        beta_poloidal: [~] :term:`glossary link<beta_poloidal>`
        vertical_magnetic_field_equation: [~] :term:`glossary link<vertical_magnetic_field_equation>`
        surface_inductance_coefficients: [~] :term:`glossary link<surface_inductance_coefficients>`
        internal_inductivity: [~] :term:`glossary<internal_inductivity>`

    Returns:
        :term:`internal_inductivity`,
        :term:`internal_inductance`,
        :term:`external_inductance`,
        :term:`vertical_field_mutual_inductance`,
        :term:`vertical_magnetic_field`
    """
    if internal_inductivity is None:
        internal_inductivity = formulas.calc_internal_inductivity(
            plasma_current, major_radius, magnetic_field_on_axis, minor_radius, safety_factor_on_axis
        )

    internal_inductance = formulas.calc_internal_inductance(
        major_radius, internal_inductivity, plasma_volume, poloidal_circumference, internal_inductance_geometry
    )
    external_inductance = formulas.calc_external_inductance(
        inverse_aspect_ratio, areal_elongation, beta_poloidal, major_radius, internal_inductivity, surface_inductance_coefficients
    )
    vertical_field_mutual_inductance = formulas.calc_vertical_field_mutual_inductance(
        inverse_aspect_ratio, areal_elongation, surface_inductance_coefficients
    )
    invmu_0_dLedR = formulas.inductances.calc_invmu_0_dLedR(
        inverse_aspect_ratio,
        areal_elongation,
        beta_poloidal,
        internal_inductivity,
        external_inductance,
        major_radius,
        surface_inductance_coefficients,
    )
    vertical_magnetic_field = formulas.calc_vertical_magnetic_field(
        inverse_aspect_ratio,
        areal_elongation,
        beta_poloidal,
        internal_inductivity,
        external_inductance,
        major_radius,
        plasma_current,
        invmu_0_dLedR,
        vertical_magnetic_field_equation,
        surface_inductance_coefficients,
    )

    local_vars = locals()
    return {key: convert_to_default_units(local_vars[key], key) for key in RETURN_KEYS}


calc_inductances = Algorithm(
    function=run_calc_inductances,
    return_keys=RETURN_KEYS,
)
