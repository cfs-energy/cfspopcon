"""Resistive flux consumption, inductive flux consumption (internally and externally on the plasma surface) during purely ohmic ramp-up."""
import numpy as np
from scipy import constants  # type: ignore[import]

from cfspopcon.named_options import InternalInductanceGeometry, SurfaceInductanceCoeffs, VertMagneticFieldEq

from ..unit_handling import Unitfull, ureg, wraps_ufunc


def select_coeffs(surface_inductance_coefficients: SurfaceInductanceCoeffs) -> tuple:
    """Choose which coefficients you want to use for the external flux calculation.

    1. Barr's Coefficients cite:`Barr_2018`.
    2. Hirshman's Coefficients cite:'hirshman'.

    """
    if surface_inductance_coefficients == SurfaceInductanceCoeffs.Barr:
        a = np.array([1.438, 2.139, 9.387, -1.939])
        b = np.array([0.149, 1.068, -6.216, 4.126])
        c = np.array([-0.293, -0.349, 0.098])
        d = np.array([0.003, 0.334, -2.018])
        e = np.array([0.080, -0.260, -0.267, 1.135])
    elif surface_inductance_coefficients == SurfaceInductanceCoeffs.Hirshman:
        a = np.array([1.81, 2.05, 9.25, -1.21])
        b = np.array([0.73, 2, -6.00, 3.70])
        c = np.array([0.98, 0.49, 1.47])
        d = np.array([0.25, 0.84, -1.44])
        e = np.array([0, 0, 0, 0])  # N/A
    else:
        raise NotImplementedError(f"Unrecognised SurfaceInductanceCoeffs option {surface_inductance_coefficients.name}")

    return a, b, c, d, e


@wraps_ufunc(
    input_units=dict(
        plasma_current=ureg.A,
        major_radius=ureg.m,
        magnetic_field_on_axis=ureg.T,
        minor_radius=ureg.m,
        safety_factor_on_axis=ureg.dimensionless,
    ),
    return_units=dict(internal_inductivity=ureg.dimensionless),
)
def calc_internal_inductivity(
    plasma_current: float, major_radius: float, magnetic_field_on_axis: float, minor_radius: float, safety_factor_on_axis: float = 1
) -> Unitfull:
    """Calculate the normalized internal inductance for an assumed circular plasma cross-section.

    Tokamaks (pg.120): Physics :cite:`wesson_tokamaks_2011`

    Args:
        plasma_current: [A] :term:`glossary link<plasma_current>`
        major_radius: [m] :term:`glossary link<major_radius>`
        magnetic_field_on_axis: [T] :term:`glossary link<magnetic_field_on_axis>`
        minor_radius: [m] :term:`glossary link<minor_radius>`
        safety_factor_on_axis: [~] :term:`glossary link<safety_factor_on_axis>`

    Returns:
        [~] :term:`internal_inductivity`
    """
    q_a = float(
        2 * np.pi * minor_radius**2 * magnetic_field_on_axis / (constants.mu_0 * major_radius * plasma_current)
    )  # safety-factor at edge of plasma assuming circular cross-section

    return float(np.log(1.65 + 0.89 * ((q_a / safety_factor_on_axis) - 1)))


@wraps_ufunc(
    input_units=dict(
        major_radius=ureg.m,
        internal_inductivity=ureg.dimensionless,
        plasma_volume=ureg.meter**3,
        poloidal_circumference=ureg.meter,
        internal_inductance_geometry=None,
    ),
    return_units=dict(internal_inductance=ureg.henry),
    pass_as_kwargs=("internal_inductance_geometry",),
)
def calc_internal_inductance(
    major_radius: float,
    internal_inductivity: float,
    plasma_volume: float = 0 * ureg.meter**3,
    poloidal_circumference: float = 0 * ureg.meter,
    internal_inductance_geometry: InternalInductanceGeometry = InternalInductanceGeometry.Cylindrical,
) -> Unitfull:
    """Calculate the internal inductance of the plasma (assuming a circular cross-section).

    A power-balance model for local helicity injection startup in a spherical tokamak :cite:`Barr_2018`

    Args:
        major_radius: [m] :term:`glossary link<major_radius>`
        internal_inductivity: [~] :term:`glossary link<internal_inductivity>`
        plasma_volume: [m^3] : term:`glossary link<plasma_volume>`
        poloidal_circumference: [m] :term:`glossary link<poloidal_circumference>`
        internal_inductance_geometry : [] :term:`glossary link<internal_inductance_geometry>`

    Returns:
        [henry] :term:`internal_inductance`
    """
    internal_inductance = constants.mu_0 * major_radius * internal_inductivity / 2

    if internal_inductance_geometry == InternalInductanceGeometry.NonCylindrical:
        internal_inductance = constants.mu_0 * internal_inductivity * plasma_volume / (poloidal_circumference**2)

    return float(internal_inductance)


@wraps_ufunc(
    input_units=dict(
        inverse_aspect_ratio=ureg.dimensionless,
        areal_elongation=ureg.dimensionless,
        beta_poloidal=ureg.dimensionless,
        major_radius=ureg.m,
        internal_inductivity=ureg.dimensionless,
        surface_inductance_coefficients=None,
    ),
    return_units=dict(external_inductance=ureg.henry),
    pass_as_kwargs=("surface_inductance_coefficients",),
)
def calc_external_inductance(
    inverse_aspect_ratio: float,
    areal_elongation: float,
    beta_poloidal: float,
    major_radius: float,
    internal_inductivity: float,
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
    return float(
        constants.mu_0
        * major_radius
        * calc_fa.__wrapped__(inverse_aspect_ratio, beta_poloidal, internal_inductivity, surface_inductance_coefficients)
        * (1 - inverse_aspect_ratio)
        / ((1 - inverse_aspect_ratio) + areal_elongation * calc_fb.__wrapped__(inverse_aspect_ratio, surface_inductance_coefficients))
    )


@wraps_ufunc(
    input_units=dict(inverse_aspect_ratio=ureg.dimensionless, areal_elongation=ureg.dimensionless, surface_inductance_coefficients=None),
    return_units=dict(Mv=ureg.dimensionless),
    pass_as_kwargs=("surface_inductance_coefficients",),
)
def calc_vertical_field_mutual_inductance(
    inverse_aspect_ratio: float, areal_elongation: float, surface_inductance_coefficients: SurfaceInductanceCoeffs
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
    return float(
        (1 - inverse_aspect_ratio) ** 2
        / (
            (1 - inverse_aspect_ratio) ** 2 * calc_fc.unitless_func(inverse_aspect_ratio, surface_inductance_coefficients)
            + calc_fd.unitless_func(inverse_aspect_ratio, areal_elongation, surface_inductance_coefficients) * np.sqrt(areal_elongation)
        )
    )


@wraps_ufunc(
    input_units=dict(
        inverse_aspect_ratio=ureg.dimensionless,
        areal_elongation=ureg.dimensionless,
        beta_poloidal=ureg.dimensionless,
        internal_inductivity=ureg.dimensionless,
        external_inductance=ureg.henry,
        major_radius=ureg.m,
        surface_inductance_coefficients=None,
    ),
    return_units=dict(invmu_0_dLedR=ureg.dimensionless),
    pass_as_kwargs=("surface_inductance_coefficients",),
)
def calc_invmu_0_dLedR(
    inverse_aspect_ratio: float,
    areal_elongation: float,
    beta_poloidal: float,
    internal_inductivity: float,
    external_inductance: float,
    major_radius: float,
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
    invmu_0_dLedR = float(
        (1 / constants.mu_0)
        * (
            constants.mu_0
            * inverse_aspect_ratio
            * (1 - inverse_aspect_ratio)
            * calc_fa.unitless_func(inverse_aspect_ratio, beta_poloidal, internal_inductivity, surface_inductance_coefficients)
            * calc_fh.unitless_func(inverse_aspect_ratio, areal_elongation, surface_inductance_coefficients)
            / (
                (
                    (1 - inverse_aspect_ratio)
                    + areal_elongation * calc_fb.unitless_func(inverse_aspect_ratio, surface_inductance_coefficients)
                )
                ** 2
            )
            - constants.mu_0
            * inverse_aspect_ratio
            * (1 - inverse_aspect_ratio)
            * calc_fg.unitless_func(inverse_aspect_ratio, beta_poloidal, internal_inductivity, surface_inductance_coefficients)
            / ((1 - inverse_aspect_ratio) + areal_elongation * calc_fb.unitless_func(inverse_aspect_ratio, surface_inductance_coefficients))
            + (inverse_aspect_ratio)
            * constants.mu_0
            * calc_fa.unitless_func(inverse_aspect_ratio, beta_poloidal, internal_inductivity, surface_inductance_coefficients)
            / ((1 - inverse_aspect_ratio) + areal_elongation * calc_fb.unitless_func(inverse_aspect_ratio, surface_inductance_coefficients))
            + external_inductance / major_radius
        )
    )

    return invmu_0_dLedR


@wraps_ufunc(
    input_units=dict(
        inverse_aspect_ratio=ureg.dimensionless,
        areal_elongation=ureg.dimensionless,
        beta_poloidal=ureg.dimensionless,
        internal_inductivity=ureg.dimensionless,
        external_inductance=ureg.henry,
        major_radius=ureg.m,
        plasma_current=ureg.A,
        invmu_0_dLedR=ureg.dimensionless,
        vertical_magnetic_field_equation=None,
        surface_inductance_coefficients=None,
    ),
    return_units=dict(vertical_magnetic_field=ureg.T),
    pass_as_kwargs=("surface_inductance_coefficients",),
)
def calc_vertical_magnetic_field(
    inverse_aspect_ratio: float,
    areal_elongation: float,
    beta_poloidal: float,
    internal_inductivity: float,
    external_inductance: float,
    major_radius: float,
    plasma_current: float,
    invmu_0_dLedR: float = 0,
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
    if invmu_0_dLedR != 0:
        vertical_magnetic_field = float(
            constants.mu_0
            * plasma_current
            * (1 / (4 * np.pi * major_radius))
            * (invmu_0_dLedR + (beta_poloidal + (internal_inductivity / 2)) - (1 / 2))
        )
    if vertical_magnetic_field_equation == VertMagneticFieldEq.MgnticFsionEnrgyFrmlry:
        vertical_magnetic_field = float(
            constants.mu_0
            * plasma_current
            * (1 / (4 * np.pi * major_radius))
            * (np.log(8 / inverse_aspect_ratio) + beta_poloidal + (internal_inductivity / 2) - 1.5)
        )
    elif vertical_magnetic_field_equation == VertMagneticFieldEq.Mit_and_Taka_Eq13:
        vertical_magnetic_field = float(
            constants.mu_0
            * plasma_current
            * (1 / (4 * np.pi * major_radius))
            * (
                np.log(8 / (inverse_aspect_ratio * (np.sqrt((1 + areal_elongation**2) / 2))))
                + beta_poloidal
                + (internal_inductivity / 2)
                - 1.5
            )
        )
    elif vertical_magnetic_field_equation == VertMagneticFieldEq.Jean:
        vertical_magnetic_field = float(
            constants.mu_0
            * plasma_current
            * (1 / (4 * np.pi * major_radius))
            * (np.log(8 / (inverse_aspect_ratio * (np.sqrt(areal_elongation)))) + beta_poloidal + (internal_inductivity / 2) - 1.5)
        )

    return vertical_magnetic_field


### ANALYTIC FUNCTIONS FOR PLASMA EXTERNAL INDUCTANCE AND VERTICAL FIELD ###


@wraps_ufunc(
    input_units=dict(inverse_aspect_ratio=ureg.dimensionless, surface_inductance_coefficients=None),
    return_units=dict(func_term=ureg.dimensionless),
    pass_as_kwargs=("surface_inductance_coefficients",),
)
def calc_fa_Sums_Na(inverse_aspect_ratio: float, surface_inductance_coefficients: SurfaceInductanceCoeffs) -> Unitfull:
    """Calculate a sum for eq. 17 on page 6 in :cite:`Barr_2018`.

    A power-balance model for local helicity injection startup in a spherical tokamak :cite:`Barr_2018`
    NOTE: Default values for for the coefficients 'N[a,d]' and '[a,e]' are taken from `Barr_2018` which are obtained
    from fitting them to model flux_PF and flux_Le obtained from over 330 model equilibria spanning 0<=delta<=0.5 whereas
    SPARC is projected to have delta95 = 0.54

    Args:
        inverse_aspect_ratio: [~] :term:`glossary link<inverse_aspect_ratio>`
        surface_inductance_coefficients: [~] :term:`glossary link<surface_inductance_coefficients>`

    Returns:
         functional term [~]
    """
    Na = 4
    coeffs = select_coeffs(surface_inductance_coefficients)
    a = coeffs[0]

    sum1 = 0
    sum2 = 0
    n = np.arange(Na // 2)
    m = n + 1
    sum1 = np.sum(a[n] * (np.sqrt(inverse_aspect_ratio)) ** m)
    sum2 = np.sum(a[(Na // 2) + n] * (np.sqrt(inverse_aspect_ratio)) ** m)
    return float(sum1), float(sum2)


@wraps_ufunc(
    input_units=dict(inverse_aspect_ratio=ureg.dimensionless, surface_inductance_coefficients=None),
    return_units=dict(func_term=ureg.dimensionless),
    pass_as_kwargs=("surface_inductance_coefficients",),
)
def calc_fa_Sum_Ne(inverse_aspect_ratio: float, surface_inductance_coefficients: SurfaceInductanceCoeffs) -> Unitfull:
    """Calculate a sum for eq. 17 on page 6 in :cite:`Barr_2018`.

    A power-balance model for local helicity injection startup in a spherical tokamak :cite:`Barr_2018`

    Args:
        inverse_aspect_ratio: [~] :term:`glossary link<inverse_aspect_ratio>`
        surface_inductance_coefficients: [~] :term:`glossary link<surface_inductance_coefficients>`

    Returns:
         functional term [~]
    """
    Ne = 4
    coeffs = select_coeffs(surface_inductance_coefficients)
    e = coeffs[4]

    n = np.arange(0, Ne)
    m = n + 1
    return float(np.sum(e[n] * np.sqrt(inverse_aspect_ratio) ** m))


@wraps_ufunc(
    input_units=dict(inverse_aspect_ratio=ureg.dimensionless, surface_inductance_coefficients=None),
    return_units=dict(func_term=ureg.dimensionless),
    pass_as_kwargs=("surface_inductance_coefficients",),
)
def calc_fb_Sum_Nb(inverse_aspect_ratio: float, surface_inductance_coefficients: SurfaceInductanceCoeffs) -> Unitfull:
    """Calculate the sum for eq. 18 on page 6 in :cite:`Barr_2018`.

    A power-balance model for local helicity injection startup in a spherical tokamak :cite:`Barr_2018`


    Args:
        inverse_aspect_ratio: [~] :term:`glossary link<inverse_aspect_ratio>`
        surface_inductance_coefficients: [~] :term:`glossary link<surface_inductance_coefficients>`

    Returns:
         functional term [~]
    """
    Nb = 4
    coeffs = select_coeffs(surface_inductance_coefficients)
    b = coeffs[1]

    n = np.arange(1, Nb)
    m = n + 1
    return float(np.sum(b[n] * inverse_aspect_ratio ** (2 + m)))


@wraps_ufunc(
    input_units=dict(inverse_aspect_ratio=ureg.dimensionless, surface_inductance_coefficients=None),
    return_units=dict(func_term=ureg.dimensionless),
    pass_as_kwargs=("surface_inductance_coefficients",),
)
def calc_fc_Sum_Nc(inverse_aspect_ratio: float, surface_inductance_coefficients: SurfaceInductanceCoeffs) -> Unitfull:
    """Calculate the sum for eq. 18 on page 6 in :cite:`Barr_2018`.

    A power-balance model for local helicity injection startup in a spherical tokamak :cite:`Barr_2018`

    Args:
        inverse_aspect_ratio: [~] :term:`glossary link<inverse_aspect_ratio>`
        surface_inductance_coefficients: [~] :term:`glossary link<surface_inductance_coefficients>`

    Returns:
         functional term [~]
    """
    Nc = 3
    coeffs = select_coeffs(surface_inductance_coefficients)
    c = coeffs[2]

    n = np.arange(Nc)
    m = n + 1
    return float(np.sum(c[n] * inverse_aspect_ratio ** (2 * m)))


@wraps_ufunc(
    input_units=dict(inverse_aspect_ratio=ureg.dimensionless, surface_inductance_coefficients=None),
    return_units=dict(func_term=ureg.dimensionless),
    pass_as_kwargs=("surface_inductance_coefficients",),
)
def calc_fd_Sum_Nd(inverse_aspect_ratio: float, surface_inductance_coefficients: SurfaceInductanceCoeffs) -> Unitfull:
    """Calculate the sum for eq. 20 on page 6 in :cite:`Barr_2018`.

    A power-balance model for local helicity injection startup in a spherical tokamak :cite:`Barr_2018`

    Args:
        inverse_aspect_ratio: [~] :term:`glossary link<inverse_aspect_ratio>`
        surface_inductance_coefficients: [~] :term:`glossary link<surface_inductance_coefficients>`

    Returns:
         functional term [~]
    """
    Nd = 3
    coeffs = select_coeffs(surface_inductance_coefficients)
    d = coeffs[3]

    n = np.arange(Nd - 1)
    m = n + 2
    return float(np.sum(d[n + 1] * inverse_aspect_ratio ** (m - 1)))


@wraps_ufunc(
    input_units=dict(inverse_aspect_ratio=ureg.dimensionless, surface_inductance_coefficients=None),
    return_units=dict(func_term=ureg.dimensionless),
    pass_as_kwargs=("surface_inductance_coefficients",),
)
def calc_fg_Sums_Na(inverse_aspect_ratio: float, surface_inductance_coefficients: SurfaceInductanceCoeffs) -> Unitfull:
    """Calculate sums for eq. 22 on page 6 in :cite:`Barr_2018`.

    A power-balance model for local helicity injection startup in a spherical tokamak :cite:`Barr_2018`

    Args:
        inverse_aspect_ratio: [~] :term:`glossary link<inverse_aspect_ratio>`
        surface_inductance_coefficients: [~] :term:`glossary link<surface_inductance_coefficients>`

    Returns:
         functional term [~]
    """
    coeffs = select_coeffs(surface_inductance_coefficients)
    a = coeffs[0]

    sum1 = (1 / 2) * a[0] * (np.sqrt(inverse_aspect_ratio) ** -1) + a[1]
    sum2 = (a[0] + (1 / 2) * a[2]) * (1 / np.sqrt(inverse_aspect_ratio)) + (a[1] + a[3])
    return float(sum1), float(sum2)


@wraps_ufunc(
    input_units=dict(inverse_aspect_ratio=ureg.dimensionless, surface_inductance_coefficients=None),
    return_units=dict(func_term=ureg.dimensionless),
    pass_as_kwargs=("surface_inductance_coefficients",),
)
def calc_fg_Sum_Ce(inverse_aspect_ratio: float, surface_inductance_coefficients: SurfaceInductanceCoeffs) -> Unitfull:
    """Calculate a sum for eq. 22 on page 6 in :cite:`Barr_2018`.

    A power-balance model for local helicity injection startup in a spherical tokamak :cite:`Barr_2018`

    Args:
        inverse_aspect_ratio: [~] :term:`glossary link<inverse_aspect_ratio>`
        surface_inductance_coefficients: [~] :term:`glossary link<surface_inductance_coefficients>`

    Returns:
         functional term [~]
    """
    Ce = 4
    coeffs = select_coeffs(surface_inductance_coefficients)
    e = coeffs[4]

    n = np.arange(Ce)
    m = n + 1
    return float(np.sum((m / 2) * e[n] * np.sqrt(inverse_aspect_ratio) ** (m - 1)))


@wraps_ufunc(
    input_units=dict(inverse_aspect_ratio=ureg.dimensionless, surface_inductance_coefficients=None),
    return_units=dict(func_term=ureg.dimensionless),
    pass_as_kwargs=("surface_inductance_coefficients",),
)
def calc_fh_Sum_Cb(inverse_aspect_ratio: float, surface_inductance_coefficients: SurfaceInductanceCoeffs) -> Unitfull:
    """Calculate a sum for eq. 23 on page 6 in :cite:`Barr_2018`.

    A power-balance model for local helicity injection startup in a spherical tokamak :cite:`Barr_2018`

    Args:
        inverse_aspect_ratio: [~] :term:`glossary link<inverse_aspect_ratio>`
        surface_inductance_coefficients: [~] :term:`glossary link<surface_inductance_coefficients>`

    Returns:
         functional term [~]
    """
    Cb = 4
    coeffs = select_coeffs(surface_inductance_coefficients)
    b = coeffs[1]

    n = np.arange(1, Cb)
    m = n + 1
    return float(np.sum((m + 2.5) * b[n] * inverse_aspect_ratio ** (m + 2)))


@wraps_ufunc(
    input_units=dict(
        inverse_aspect_ratio=ureg.dimensionless,
        beta_poloidal=ureg.dimensionless,
        internal_inductivity=ureg.dimensionless,
        surface_inductance_coefficients=None,
    ),
    return_units=dict(func=ureg.dimensionless),
    pass_as_kwargs=("surface_inductance_coefficients",),
)
def calc_fa(
    inverse_aspect_ratio: float, beta_poloidal: float, internal_inductivity: float, surface_inductance_coefficients: SurfaceInductanceCoeffs
) -> float:
    """Calculate eq. 17 on page 6 in :cite:`Barr_2018`.

    A power-balance model for local helicity injection startup in a spherical tokamak :cite:`Barr_2018`

    Args:
        inverse_aspect_ratio: [~] :term:`glossary link<inverse_aspect_ratio>`
        beta_poloidal: [~] :term:`glossary link<beta_poloidal>`
        internal_inductivity: [~] :term:`glossary link<internal_inductivity>`
        surface_inductance_coefficients: [~] :term:`glossary link<surface_inductance_coefficients>`

    Returns:
         functional term [~]
    """
    fa_Sums_ = calc_fa_Sums_Na.__wrapped__(inverse_aspect_ratio, surface_inductance_coefficients)

    return float(
        ((1 + fa_Sums_[0]) * np.log(8 / inverse_aspect_ratio))
        - (2 + fa_Sums_[1])
        + (beta_poloidal + internal_inductivity / 2) * calc_fa_Sum_Ne.unitless_func(inverse_aspect_ratio, surface_inductance_coefficients)
    )


@wraps_ufunc(
    input_units=dict(inverse_aspect_ratio=ureg.dimensionless, surface_inductance_coefficients=None),
    return_units=dict(func_term=ureg.dimensionless),
    pass_as_kwargs=("surface_inductance_coefficients",),
)
def calc_fb(inverse_aspect_ratio: float, surface_inductance_coefficients: SurfaceInductanceCoeffs) -> float:
    """Calculate eq. 18 on page 6 in :cite:`Barr_2018`.

    A power-balance model for local helicity injection startup in a spherical tokamak :cite:`Barr_2018`

    Args:
        inverse_aspect_ratio: [~] :term:`glossary link<inverse_aspect_ratio>`
        surface_inductance_coefficients: [~] :term:`glossary link<surface_inductance_coefficients>`

    Returns:
         functional term [~]
    """
    coeffs = select_coeffs(surface_inductance_coefficients)
    b = coeffs[1]

    return float(
        b[0] * np.sqrt(inverse_aspect_ratio) * (1 + calc_fb_Sum_Nb.unitless_func(inverse_aspect_ratio, surface_inductance_coefficients))
    )


@wraps_ufunc(
    input_units=dict(inverse_aspect_ratio=ureg.dimensionless, surface_inductance_coefficients=None),
    return_units=dict(func_term=ureg.dimensionless),
    pass_as_kwargs=("surface_inductance_coefficients",),
)
def calc_fc(inverse_aspect_ratio: float, surface_inductance_coefficients: SurfaceInductanceCoeffs) -> float:
    """Calculate eq. 19 on page 6 in :cite:`Barr_2018`.

    A power-balance model for local helicity injection startup in a spherical tokamak :cite:`Barr_2018`

    Args:
        inverse_aspect_ratio: [~] :term:`glossary link<inverse_aspect_ratio>`
        surface_inductance_coefficients: [~] :term:`glossary link<surface_inductance_coefficients>`

    Returns:
         functional term [~]
    """
    fc = float(1 + calc_fc_Sum_Nc.unitless_func(inverse_aspect_ratio, surface_inductance_coefficients))

    return fc


@wraps_ufunc(
    input_units=dict(inverse_aspect_ratio=ureg.dimensionless, areal_elongation=ureg.dimensionless, surface_inductance_coefficients=None),
    return_units=dict(func_term=ureg.dimensionless),
    pass_as_kwargs=("surface_inductance_coefficients",),
)
def calc_fd(inverse_aspect_ratio: float, areal_elongation: float, surface_inductance_coefficients: SurfaceInductanceCoeffs) -> float:
    """Calculate eq. 20 on page 6 in :cite:`Barr_2018`.

    A power-balance model for local helicity injection startup in a spherical tokamak :cite:`Barr_2018`

    Args:
        inverse_aspect_ratio: [~] :term:`glossary link<inverse_aspect_ratio>`
        areal_elongation: [~] :term:`glossary link<areal_elongation>`
        surface_inductance_coefficients: [~] :term:`glossary link<surface_inductance_coefficients>`

    Returns:
         functional term [~]
    """
    coeffs = select_coeffs(surface_inductance_coefficients)
    d = coeffs[3]
    fd = float(d[0] * inverse_aspect_ratio * (1 + calc_fd_Sum_Nd.unitless_func(inverse_aspect_ratio, surface_inductance_coefficients)))

    return fd


@wraps_ufunc(
    input_units=dict(
        inverse_aspect_ratio=ureg.dimensionless,
        beta_poloidal=ureg.dimensionless,
        internal_inductivity=ureg.dimensionless,
        surface_inductance_coefficients=None,
    ),
    return_units=dict(func_term=ureg.dimensionless),
    pass_as_kwargs=("surface_inductance_coefficients",),
)
def calc_fg(
    inverse_aspect_ratio: float, beta_poloidal: float, internal_inductivity: float, surface_inductance_coefficients: SurfaceInductanceCoeffs
) -> float:
    """Calculate eq. 22 on page 6 in :cite:`Barr_2018`.

    A power-balance model for local helicity injection startup in a spherical tokamak :cite:`Barr_2018`

    Args:
        inverse_aspect_ratio: [~] :term:`glossary link<inverse_aspect_ratio>`
        beta_poloidal: [~] :term:`glossary link<beta_poloidal>`
        internal_inductivity: [~] :term:`glossary link<internal_inductivity>`
        surface_inductance_coefficients: [~] :term:`glossary link<surface_inductance_coefficients>`

    Returns:
         functional term [~]
    """
    fg_Sums_ = calc_fg_Sums_Na.__wrapped__(inverse_aspect_ratio, surface_inductance_coefficients)

    return float(
        -(1 / inverse_aspect_ratio)
        + np.log(8 / inverse_aspect_ratio) * fg_Sums_[0]
        - fg_Sums_[1]
        + (beta_poloidal + (internal_inductivity / 2)) * calc_fg_Sum_Ce.__wrapped__(inverse_aspect_ratio, surface_inductance_coefficients)
    )


@wraps_ufunc(
    input_units=dict(inverse_aspect_ratio=ureg.dimensionless, areal_elongation=ureg.dimensionless, surface_inductance_coefficients=None),
    return_units=dict(func_term=ureg.dimensionless),
    pass_as_kwargs=("surface_inductance_coefficients",),
)
def calc_fh(inverse_aspect_ratio: Unitfull, areal_elongation: Unitfull, surface_inductance_coefficients: SurfaceInductanceCoeffs) -> float:
    """Calculate eq. 23 on page 6 in :cite:`Barr_2018`.

    A power-balance model for local helicity injection startup in a spherical tokamak :cite:`Barr_2018`

    Args:
        inverse_aspect_ratio: [~] :term:`glossary link<inverse_aspect_ratio>`
        areal_elongation: [~] :term:`glossary link<areal_elongation>`
        surface_inductance_coefficients: [~] :term:`glossary link<surface_inductance_coefficients>`

    Returns:
         functional term [~]
    """
    coeffs = select_coeffs(surface_inductance_coefficients)
    b = coeffs[1]

    return float(
        -1
        + ((areal_elongation * b[0]) / np.sqrt(inverse_aspect_ratio))
        * (1 / 2 + calc_fh_Sum_Cb.__wrapped__(inverse_aspect_ratio, surface_inductance_coefficients))
    )
