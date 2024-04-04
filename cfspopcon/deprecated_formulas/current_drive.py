"""Ohmic and bootstrap plasma current, loop resistivity & voltage, and current relaxation time."""
from ..unit_handling import Unitfull, ureg, wraps_ufunc


def calc_f_shaping(inverse_aspect_ratio: Unitfull, areal_elongation: Unitfull, triangularity_psi95: Unitfull) -> Unitfull:
    """Calculate the shaping function.

    Equation A11 from ITER Physics Basis Ch. 1. Eqn. A-11 :cite:`editors_iter_1999`
    See following discussion for how this function is used.
    q_95 = 5 * minor_radius^2 * magnetic_field_on_axis / (R * plasma_current) f_shaping

    Args:
        inverse_aspect_ratio: [~] :term:`glossary link<inverse_aspect_ratio>`
        areal_elongation: [~] :term:`glossary link<areal_elongation>`
        triangularity_psi95: [~] :term:`glossary link<triangularity_psi95>`

    Returns:
        :term:`f_shaping` [~]
    """
    return ((1.0 + areal_elongation**2.0 * (1.0 + 2.0 * triangularity_psi95**2.0 - 1.2 * triangularity_psi95**3.0)) / 2.0) * (
        (1.17 - 0.65 * inverse_aspect_ratio) / (1.0 - inverse_aspect_ratio**2.0) ** 2.0
    )


@wraps_ufunc(
    input_units=dict(
        magnetic_field_on_axis=ureg.T,
        major_radius=ureg.m,
        inverse_aspect_ratio=ureg.dimensionless,
        q_star=ureg.dimensionless,
        f_shaping=ureg.dimensionless,
    ),
    return_units=dict(plasma_current=ureg.MA),
)
def calc_plasma_current(
    magnetic_field_on_axis: float, major_radius: float, inverse_aspect_ratio: float, q_star: float, f_shaping: float
) -> float:
    """Calculate the plasma current in mega-amperes.

    Updated formula from ITER Physics Basis Ch. 1. :cite:`editors_iter_1999`

    Args:
        magnetic_field_on_axis: [T] :term:`glossary link<magnetic_field_on_axis>`
        major_radius: [m] :term:`glossary link<major_radius>`
        inverse_aspect_ratio: [~] :term:`glossary link<inverse_aspect_ratio>`
        q_star: [~] :term:`glossary link<q_star>`
        f_shaping: [~] :term:`glossary link<f_shaping>`

    Returns:
        :term:`plasma_current` [MA]
    """
    return float(5.0 * ((inverse_aspect_ratio * major_radius) ** 2.0) * (magnetic_field_on_axis / (q_star * major_radius)) * f_shaping)


@wraps_ufunc(
    input_units=dict(
        magnetic_field_on_axis=ureg.T,
        major_radius=ureg.m,
        inverse_aspect_ratio=ureg.dimensionless,
        plasma_current=ureg.MA,
        f_shaping=ureg.dimensionless,
    ),
    return_units=dict(q_star=ureg.dimensionless),
)
def calc_q_star(
    magnetic_field_on_axis: float, major_radius: float, inverse_aspect_ratio: float, plasma_current: float, f_shaping: float
) -> float:
    """Calculate an analytical estimate for the edge safety factor q_star.

    Updated formula from ITER Physics Basis Ch. 1. :cite:`editors_iter_1999`

    Args:
        magnetic_field_on_axis: [T] :term:`glossary link<magnetic_field_on_axis>`
        major_radius: [m] :term:`glossary link<major_radius>`
        inverse_aspect_ratio: [~] :term:`glossary link<inverse_aspect_ratio>`
        plasma_current: [MA] :term:`glossary link<plasma_current>`
        f_shaping: [~] :term:`glossary link<f_shaping>`

    Returns:
        :term:`plasma_current` [MA]
    """
    return float(5.0 * (inverse_aspect_ratio * major_radius) ** 2.0 * magnetic_field_on_axis / (plasma_current * major_radius) * f_shaping)


def calc_ohmic_power(inductive_plasma_current: Unitfull, loop_voltage: Unitfull) -> Unitfull:
    """Calculate the Ohmic heating power.

    Args:
        inductive_plasma_current: [MA] :term:`glossary link<inductive_plasma_current>`
        loop_voltage: [V] :term:`glossary link<loop_voltage>`

    Returns:
        :term:`P_Ohmic` [MW]
    """
    return inductive_plasma_current * loop_voltage


@wraps_ufunc(input_units=dict(average_electron_temp=ureg.keV), return_units=dict(spitzer_resistivity=ureg.ohm * ureg.m))
def calc_Spitzer_loop_resistivity(average_electron_temp: float) -> float:
    """Calculate the parallel Spitzer loop resistivity assuming the Coulomb logarithm = 17 and Z=1.

    Resistivity from Wesson 2.16.2 :cite:`wesson_tokamaks_2011`

    Args:
        average_electron_temp: [keV] :term:`glossary link<average_electron_temp>`

    Returns:
        :term:`spitzer_resistivity` [Ohm-m]
    """
    return float((2.8e-8) * (average_electron_temp ** (-1.5)))


def calc_resistivity_trapped_enhancement(inverse_aspect_ratio: Unitfull, definition: int = 3) -> Unitfull:
    """Calculate the enhancement of the plasma resistivity due to trapped particles.

    Definition 1 is the denominator of eta_n (neoclassical resistivity) on p801 of Wesson :cite:`wesson_tokamaks_2011`

    Args:
        inverse_aspect_ratio: [~] :term:`glossary link<inverse_aspect_ratio>`
        definition: [~] choice of [1,2,3] to specify which definition to use

    Returns:
        :term:`trapped_particle_fraction` [~]

    Raises:
        NotImplementedError: if definition doesn't match an implementation
    """
    if definition == 1:
        trapped_particle_fraction = 1 / ((1.0 - (inverse_aspect_ratio**0.5)) ** 2.0)  # pragma: nocover
    elif definition == 2:
        trapped_particle_fraction = 2 / (1.0 - 1.31 * (inverse_aspect_ratio**0.5) + 0.46 * inverse_aspect_ratio)  # pragma: nocover
    elif definition == 3:
        trapped_particle_fraction = 0.609 / (0.609 - 0.785 * (inverse_aspect_ratio**0.5) + 0.269 * inverse_aspect_ratio)
    else:
        raise NotImplementedError(f"No implementation {definition} for calc_resistivity_trapped_enhancement.")  # pragma: nocover

    return trapped_particle_fraction


def calc_neoclassical_loop_resistivity(
    spitzer_resistivity: Unitfull, z_effective: Unitfull, trapped_particle_fraction: Unitfull
) -> Unitfull:
    """Calculate the neoclassical loop resistivity including impurity ions.

    Wesson Section 14.10. Impact of ion charge. Impact of dilution ~ 0.9.

    Args:
        spitzer_resistivity: [Ohm-m] :term:`glossary link<spitzer_resistivity>`
        z_effective: [~] :term:`glossary link<z_effective>`
        trapped_particle_fraction: [~] :term:`glossary link<trapped_particle_fraction>`

    Returns:
        :term:`neoclassical_loop_resistivity` [Ohm-m]
    """
    return spitzer_resistivity * z_effective * 0.9 * trapped_particle_fraction


@wraps_ufunc(
    input_units=dict(
        major_radius=ureg.m,
        inverse_aspect_ratio=ureg.dimensionless,
        areal_elongation=ureg.dimensionless,
        average_electron_temp=ureg.keV,
        z_effective=ureg.dimensionless,
    ),
    return_units=dict(current_relaxation_time=ureg.s),
)
def calc_current_relaxation_time(
    major_radius: float, inverse_aspect_ratio: float, areal_elongation: float, average_electron_temp: float, z_effective: float
) -> float:
    """Calculate the current relaxation time.

    from :cite:`Bonoli`

    Args:
        major_radius: [m] :term:`glossary link<major_radius>`
        inverse_aspect_ratio: [~] :term:`glossary link<inverse_aspect_ratio>`
        areal_elongation: [~] :term:`glossary link<areal_elongation>`
        average_electron_temp: [keV] :term:`glossary link<average_electron_temp>`
        z_effective: [~] :term:`glossary link<z_effective>`

    Returns:
        :term:`current_relaxation_time` [s]
    """
    return float(
        1.4 * ((major_radius * inverse_aspect_ratio) ** 2.0) * areal_elongation * (average_electron_temp**1.5) / z_effective
    )  # [s]


def calc_loop_voltage(
    major_radius: Unitfull,
    minor_radius: Unitfull,
    inductive_plasma_current: Unitfull,
    areal_elongation: Unitfull,
    neoclassical_loop_resistivity: Unitfull,
) -> Unitfull:
    """Calculate plasma toroidal loop voltage at flattop.

    Plasma loop voltage from Alex Creely's original work.

    Args:
        major_radius: [m] :term:`glossary link<major_radius>`
        minor_radius: [m] :term:`glossary link<minor_radius>`
        inductive_plasma_current: [MA] :term:`glossary link<inductive_plasma_current>`
        areal_elongation: [~] :term:`glossary link<areal_elongation>`
        neoclassical_loop_resistivity: [Ohm-m] :term:`glossary link<neoclassical_loop_resistivity>`

    Returns:
        :term:`loop_voltage` [V]
    """
    Iind = inductive_plasma_current  # Inductive plasma current [A]

    _term1 = 2 * major_radius / (minor_radius**2 * areal_elongation)  # Toroidal length over plasma cross-section surface area [1/m]
    return Iind * _term1 * neoclassical_loop_resistivity


def calc_bootstrap_fraction(
    ion_density_peaking: Unitfull,
    electron_density_peaking: Unitfull,
    temperature_peaking: Unitfull,
    z_effective: Unitfull,
    q_star: Unitfull,
    inverse_aspect_ratio: Unitfull,
    beta_poloidal: Unitfull,
) -> Unitfull:
    """Calculate bootstrap current fraction.

    K. Gi et al, Bootstrap current fraction scaling :cite:`gi_bootstrap_2014`
    Equation assumes q0 = 1

    Args:
        ion_density_peaking: [~] :term:`glossary link<ion_density_peaking>`
        electron_density_peaking: [~] :term:`glossary link<electron_density_peaking>`
        temperature_peaking: [~] :term:`glossary link<temperature_peaking>`
        z_effective: [~] :term:`glossary link<z_effective>`
        q_star: [~] :term:`glossary link<q_star>`
        inverse_aspect_ratio: [~] :term:`glossary link<inverse_aspect_ratio>`
        beta_poloidal: [~] :term:`glossary link<beta_poloidal>`

    Returns:
        :term:`bootstrap_fraction` [~]
    """
    nu_n = (ion_density_peaking + electron_density_peaking) / 2

    bootstrap_fraction = 0.474 * (
        (temperature_peaking - 1.0 + nu_n - 1.0) ** 0.974
        * (temperature_peaking - 1.0) ** -0.416
        * z_effective**0.178
        * q_star**-0.133
        * inverse_aspect_ratio**0.4
        * beta_poloidal
    )

    return bootstrap_fraction
