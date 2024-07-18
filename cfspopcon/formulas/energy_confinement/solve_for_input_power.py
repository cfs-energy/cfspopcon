"""Rearrange a tau_e scaling to give a value for the input power, given the stored energy."""

import numpy as np

from ...algorithm_class import Algorithm
from ...unit_handling import ureg, wraps_ufunc
from .read_energy_confinement_scalings import ConfinementScaling


@Algorithm.register_algorithm(return_keys=["energy_confinement_time", "P_in"])
@wraps_ufunc(
    return_units=dict(tau_e=ureg.s, P_tau=ureg.MW),
    input_units=dict(
        confinement_time_scalar=ureg.dimensionless,
        plasma_current=ureg.MA,
        magnetic_field_on_axis=ureg.T,
        average_electron_density=ureg.n19,
        major_radius=ureg.m,
        areal_elongation=ureg.dimensionless,
        separatrix_elongation=ureg.dimensionless,
        inverse_aspect_ratio=ureg.dimensionless,
        fuel_average_mass_number=ureg.amu,
        triangularity_psi95=ureg.dimensionless,
        separatrix_triangularity=ureg.dimensionless,
        plasma_stored_energy=ureg.MJ,
        q_star=ureg.dimensionless,
        tau_e_scaling=None,
    ),
    output_core_dims=[(), ()],
)
def solve_tau_e_scaling_for_input_power(
    confinement_time_scalar: float,
    plasma_current: float,
    magnetic_field_on_axis: float,
    average_electron_density: float,
    major_radius: float,
    areal_elongation: float,
    separatrix_elongation: float,
    inverse_aspect_ratio: float,
    fuel_average_mass_number: float,
    triangularity_psi95: float,
    separatrix_triangularity: float,
    plasma_stored_energy: float,
    q_star: float,
    tau_e_scaling: str,
) -> tuple[float, float]:
    r"""Calculate energy confinement time and input power from a tau_E scaling.

    The energy confinement time can generally be written as

    .. math::
        \tau_e = H \cdot C \cdot P_{\tau}^{\alpha_P}
        \cdot I_{MA}^{\alpha_I} \cdot B_0^{\alpha_B} \cdot \bar{n_{e,19}}^{\alpha_n} \cdot R_0^{\alpha_R}
        \cdot \kappa_A^{\alpha_{ka}} \cdot \kappa_{sep}^{\alpha_{ks}} \cdot \epsilon^{\alpha_e}
        \cdot m_i^{\alpha_A} \cdot \delta^{\alpha_d}

    We don't know :math:`P_{\tau}` in advance, so instead write

    .. math:: \tau_e = \gamma \cdot P_{\tau}^{\alpha_P}

    with

    .. math::
        \gamma = H \cdot C
        \cdot I_{MA}^{\alpha_I} \cdot B_0^{\alpha_B} \cdot \bar{n_{e,19}}^{\alpha_n} \cdot R_0^{\alpha_R}
        \cdot \kappa_A^{\alpha_{ka}} \cdot \kappa_{sep}^{\alpha_{ks}} \cdot \epsilon^{\alpha_e}
        \cdot m_i^{\alpha_A} \cdot \delta^{\alpha_d}

    We have everything that we need to evaluate :math:`\gamma`. Then, we also know that

    .. math:: \tau_e = W_p / P_{loss}

    Then, we crucially need to define what exactly we mean by the two powers that we've introduced
    (:math:`P_{\tau}` and :math:`P_{loss}`). We usually take
    :math:`P_{\tau} = P_{heating} = P_{ohmic} + P_{\alpha} + P_{aux}` and
    :math:`P_{loss}=P_{SOL} + P_{rad,core}` [Wesson definition]. From a
    simple power balance, :math:`P_{heating}=P_{loss}` and so, setting the two :math:`\tau_e` equations equal, we get that

    .. math::
        W_p / P = \gamma \cdot P^{\alpha_P}
        P^{\alpha_P + 1} = W_p / \gamma
        P = \left(W_p / \gamma \right)^{\frac{1}{\alpha_P + 1}}

    Once we have :math:`P = P_{ohmic} + P_{\alpha} + P_{aux} = P_{SOL} + P_{rad,core}`, we can calculate :math:`W_p/P`.

    However, it is also possible that the core radiated power is subtracted when calculating :math:`\tau_e`
    [Freidberg definition of :math:`W_p`], giving

    .. math::
        P_{\tau} = P_{ohmic} + P_{\alpha} + P_{aux} - P_{rad} = P_{loss} = P_{SOL}

    Then, the returned value should be interpreted as :math:`P_{SOL}`. We currently don't allow this case, since the
    general consensus is that the radiated power has not been consistently removed from scaling relationship. However,
    if you want to explore the effect of changing this assumption, you can implement a new feature.

    N.b. there are two more possible cases, where different powers are used in the two :math:`\tau_e` scalings.
    We don't allow these cases, since 1) experiments generally pick a consistent definition for :math:`P`
    and 2) this results in an equation for :math:`P` which cannot be solved analytically.

    Args:
        confinement_time_scalar: [~] confinement scaling factor
        plasma_current: [MA] :term:`glossary link<plasma_current>`
        magnetic_field_on_axis: [T] :term:`glossary link<magnetic_field_on_axis>`
        average_electron_density: [1e19 m^-3] :term:`glossary link<average_electron_density>`
        major_radius: [m] :term:`glossary link<major_radius>`
        areal_elongation: [~] :term:`glossary link<areal_elongation>`
        separatrix_elongation: [~] :term:`glossary link<separatrix_elongation>`
        inverse_aspect_ratio: [~] :term:`glossary link<inverse_aspect_ratio>`
        fuel_average_mass_number: [amu] :term:`glossary link<fuel_average_mass_number>`
        triangularity_psi95: [~] :term:`glossary link<triangularity_psi95>`
        separatrix_triangularity: [~] :term:`glossary link<separatrix_triangularity>`
        plasma_stored_energy: [MJ] :term:`glossary link<plasma_stored_energy>`
        q_star: [~] :term:`glossary link<q_star>`
        tau_e_scaling: [] :term:`glossary link<tau_e_scaling>`

    Returns:
        :term:`energy_confinement_time` [s], :term:`P_in` [MW]
    """
    scaling = ConfinementScaling.instances[tau_e_scaling]

    gamma = (
        confinement_time_scalar
        * scaling.constant
        * plasma_current**scaling.plasma_current_alpha
        * magnetic_field_on_axis**scaling.field_on_axis_alpha
        * average_electron_density**scaling.average_density_alpha
        * major_radius**scaling.major_radius_alpha
        * areal_elongation**scaling.areal_elongation_alpha
        * separatrix_elongation**scaling.separatrix_elongation_alpha
        * inverse_aspect_ratio**scaling.inverse_aspect_ratio_alpha
        * fuel_average_mass_number**scaling.mass_ratio_alpha
        * (1.0 + np.mean([triangularity_psi95, separatrix_triangularity])) ** scaling.triangularity_alpha
        * q_star**scaling.qstar_alpha
    )

    if gamma > 0.0:
        P_tau = (plasma_stored_energy / gamma) ** (1.0 / (1.0 + scaling.input_power_alpha))
    else:
        P_tau = np.inf

    tau_E = plasma_stored_energy / P_tau

    return tau_E, P_tau
