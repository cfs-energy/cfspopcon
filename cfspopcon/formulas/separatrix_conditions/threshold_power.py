"""Calculate the threshold values for the power crossing the separatrix."""

from ...algorithm_class import Algorithm
from ...named_options import ConfinementPowerScaling
from ...unit_handling import ureg, wraps_ufunc


@Algorithm.register_algorithm(return_keys=["P_LH_thresh"])
@wraps_ufunc(
    return_units=dict(P_LH_thresh=ureg.MW),
    input_units=dict(
        plasma_current=ureg.MA,
        magnetic_field_on_axis=ureg.T,
        minor_radius=ureg.m,
        major_radius=ureg.m,
        surface_area=ureg.m**2,
        fuel_average_mass_number=ureg.amu,
        average_electron_density=ureg.n19,
        confinement_power_scaling=None,
        confinement_threshold_scalar=ureg.dimensionless,
    ),
)
def calc_LH_transition_threshold_power(
    plasma_current: float,
    magnetic_field_on_axis: float,
    minor_radius: float,
    major_radius: float,
    surface_area: float,
    fuel_average_mass_number: float,
    average_electron_density: float,
    confinement_power_scaling: ConfinementPowerScaling = ConfinementPowerScaling.H_mode_Martin,
    confinement_threshold_scalar: float = 1.0,
) -> float:
    """Calculate the threshold power (crossing the separatrix) to transition into H-mode.

    From Martin NF 2008 Scaling, with mass correction :cite:`martin_power_2008`
    Added in low density branch from Ryter 2014 :cite:`Ryter_2014`

    Args:
        plasma_current: [MA] :term:`glossary link<plasma_current>`
        magnetic_field_on_axis: [T] :term:`glossary link<magnetic_field_on_axis>`
        minor_radius: [m] :term:`glossary link<minor_radius>`
        major_radius: [m] :term:`glossary link<major_radius>`
        surface_area: [m^2] :term:`glossary link<surface_area>`
        fuel_average_mass_number: [amu] :term:`glossary link<fuel_average_mass_number>`
        average_electron_density: [1e19 m^-3] :term:`glossary link<average_electron_density>`
        confinement_power_scaling: [~] :term:`glossary link<confinement_power_scaling>`
        confinement_threshold_scalar: [~] :term:`glossary link<confinement_threshold_scalar>`

    Returns:
         :term:`P_LH_thresh` [MW]
    """

    def _calc_Martin_LH_threshold(electron_density: float) -> float:
        _DEUTERIUM_MASS_NUMBER = 2.0

        return float(0.0488 * ((electron_density / 10.0) ** 0.717) * (magnetic_field_on_axis**0.803) * (surface_area**0.941)) * (
            _DEUTERIUM_MASS_NUMBER / fuel_average_mass_number
        )

    if confinement_power_scaling == ConfinementPowerScaling.H_mode_Martin:

        # Ryter 2014, equation 3
        neMin19 = (
            0.7
            * (plasma_current**0.34)
            * (magnetic_field_on_axis**0.62)
            * (minor_radius**-0.95)
            * ((major_radius / minor_radius) ** 0.4)
        )

        if average_electron_density < neMin19:
            P_LH_thresh = _calc_Martin_LH_threshold(electron_density=neMin19)
            return float(P_LH_thresh * (neMin19 / average_electron_density) ** 2.0) * confinement_threshold_scalar
        else:
            P_LH_thresh = _calc_Martin_LH_threshold(electron_density=average_electron_density)
            return P_LH_thresh * confinement_threshold_scalar

    else:
        raise NotImplementedError(
            f"No implementation for calc_LH_transition_threshold_power with confinement_power_scaling = {confinement_power_scaling} (type ({type(confinement_power_scaling)}))"
        )


calc_ratio_P_LH = Algorithm.from_single_function(
    func=lambda P_sol, P_LH_thresh: P_sol / P_LH_thresh, return_keys=["ratio_of_P_SOL_to_P_LH"], name="calc_ratio_P_LH"
)


@Algorithm.register_algorithm(return_keys=["P_LI_thresh"])
@wraps_ufunc(
    return_units=dict(P_LI_thresh=ureg.MW),
    input_units=dict(
        plasma_current=ureg.MA,
        magnetic_field_on_axis=ureg.T,
        surface_area=ureg.m**2,
        average_electron_density=ureg.n19,
        confinement_power_scaling=None,
        confinement_threshold_scalar=ureg.dimensionless,
    ),
)
def calc_LI_transition_threshold_power(
    plasma_current: float,
    magnetic_field_on_axis: float,
    surface_area: float,
    average_electron_density: float,
    confinement_power_scaling: ConfinementPowerScaling = ConfinementPowerScaling.I_mode_HubbardNF17,
    confinement_threshold_scalar: float = 1.0,
) -> float:
    """Calculate the threshold power (crossing the separatrix) to transition into I-mode.

    Note:
        "AUG" option is inspired from :cite:`ryter_i-mode_2016` and :cite:`Happel_2017`
        "HubbardNF17" uses scaling described in Fig 6 of :cite:`hubbard_threshold_2017`
        "HubbardNF12" uses scaling described in Fig 5 of ref :cite:`hubbard_threshold_2012`

    Args:
        plasma_current: [MA] :term:`glossary link<plasma_current>`
        magnetic_field_on_axis: [T] :term:`glossary link<magnetic_field_on_axis>`
        surface_area: [m^2] :term:`glossary link<surface_area>`
        average_electron_density: [1e19 m^-3] :term:`glossary link<average_electron_density>`
        confinement_power_scaling: [~] :term:`glossary link<confinement_power_scaling>`
        confinement_threshold_scalar: [~] :term:`glossary link<confinement_threshold_scalar>`

    Returns:
        :term:`P_LI_thresh` [MW]
    """

    def _calc_AUG_LI_threshold(
        average_electron_density: float, magnetic_field_on_axis: float, surface_area: float, confinement_threshold_scalar: float
    ) -> float:
        return (
            float(0.14 * (average_electron_density / 10) * (magnetic_field_on_axis / 2.4) ** 0.39 * surface_area)
            * confinement_threshold_scalar
        )

    def _calc_HubbardNF17_LI_threshold(
        average_electron_density: float, magnetic_field_on_axis: float, surface_area: float, confinement_threshold_scalar: float
    ) -> float:
        return (
            float(0.162 * (average_electron_density / 10) * (magnetic_field_on_axis**0.262) * surface_area) * confinement_threshold_scalar
        )

    def _calc_HubbardNF12_LI_threshold(
        average_electron_density: float, plasma_current: float, confinement_threshold_scalar: float
    ) -> float:
        return float(2.11 * plasma_current**0.94 * ((average_electron_density / 10.0) ** 0.65)) * confinement_threshold_scalar

    if confinement_power_scaling == ConfinementPowerScaling.I_mode_AUG:
        P_LI_thresh = _calc_AUG_LI_threshold(average_electron_density, magnetic_field_on_axis, surface_area, confinement_threshold_scalar)

    elif confinement_power_scaling == ConfinementPowerScaling.I_mode_HubbardNF17:
        P_LI_thresh = _calc_HubbardNF17_LI_threshold(
            average_electron_density, magnetic_field_on_axis, surface_area, confinement_threshold_scalar
        )

    elif confinement_power_scaling == ConfinementPowerScaling.I_mode_HubbardNF12:
        P_LI_thresh = _calc_HubbardNF12_LI_threshold(average_electron_density, plasma_current, confinement_threshold_scalar)

    else:
        raise NotImplementedError(
            f"No implementation for calc_LI_transition_threshold_power with confinement_power_scaling = {confinement_power_scaling} (type ({type(confinement_power_scaling)}))"
        )

    return float(P_LI_thresh)


calc_ratio_P_LI = Algorithm.from_single_function(
    func=lambda P_sol, P_LI_thresh: P_sol / P_LI_thresh, return_keys=["ratio_of_P_SOL_to_P_LI"], name="calc_ratio_P_LI"
)
