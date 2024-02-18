"""Calculate tau_e and P_in from a tau_e scaling and the stored energy."""
from pathlib import Path

import numpy as np
import yaml

from ...named_options import ConfinementScaling
from ...unit_handling import ureg, wraps_ufunc

# Preload the scalings (instead of doing fileio in loop)
with open(Path(__file__).parent / "tau_e_scalings.yaml") as f:
    TAU_E_SCALINGS = yaml.safe_load(f)


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
def calc_tau_e_and_P_in_from_scaling(
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
    tau_e_scaling: ConfinementScaling,
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

    If you are using a scaling where this is the case, set ``tau_e_scaling_uses_P_in=False``.
    Then, the returned value should be interpreted as :math:`P_{SOL}`.

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
        :term:`energy_confinement_time` [s], :term:`P_in` if tau_e_scaling_uses_P_in=False, else :term:`P_SOL` [MW]
    """
    scaling = TAU_E_SCALINGS[tau_e_scaling.name]

    gamma = (
        confinement_time_scalar
        * scaling["params"]["C"]
        * plasma_current ** scaling["params"]["a_I"]
        * magnetic_field_on_axis ** scaling["params"]["a_B"]
        * average_electron_density ** scaling["params"]["a_n"]
        * major_radius ** scaling["params"]["a_R"]
        * areal_elongation ** scaling["params"]["a_ka"]
        * separatrix_elongation ** scaling["params"]["a_ks"]
        * inverse_aspect_ratio ** scaling["params"]["a_e"]
        * fuel_average_mass_number ** scaling["params"]["a_A"]
        * (1.0 + np.mean([triangularity_psi95, separatrix_triangularity])) ** scaling["params"]["a_d"]
        * q_star ** scaling["params"]["a_q"]
    )
    
    if gamma > 0.0:
        P_tau = (plasma_stored_energy / gamma) ** (1.0 / (1.0 + scaling["params"]["a_P"]))
    else:
        P_tau = np.inf

    tau_E = plasma_stored_energy / P_tau

    return float(tau_E), float(P_tau)


def load_metadata(dataset: str) -> dict[str, str]:
    """Load dataset metadata from YAML file.

    Args:
        dataset: name of scaling in ./energy_confinement_time.yaml

    Returns:
        Metadata
    """
    metadata: dict[str, str] = TAU_E_SCALINGS[dataset]["metadata"]
    return metadata


def get_datasets() -> list[str]:
    """Get a list of names of valid datasets.

    Returns:
        List of names of valid datasets
    """
    datasets: list[str] = list(TAU_E_SCALINGS.keys())  # Unpack iterator to list
    return datasets
