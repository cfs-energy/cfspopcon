"""Formulas to calculate the bootstrap fraction."""
from ...algorithm_class import Algorithm
from ...unit_handling import Unitfull


@Algorithm.register_algorithm(return_keys=["bootstrap_fraction"])
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


Algorithm.from_single_function(
    func=lambda plasma_current, bootstrap_fraction: plasma_current * (1.0 - bootstrap_fraction),
    name="calc_inductive_plasma_current",
    return_keys=["inductive_plasma_current"],
)
