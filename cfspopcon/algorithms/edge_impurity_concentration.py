"""Run the two point model with a fixed sheath entrance temperature."""


from ..atomic_data import read_atomic_data
from ..formulas.scrape_off_layer_model import build_L_int_integrator, calc_required_edge_impurity_concentration
from ..named_options import Impurity
from ..unit_handling import Unitfull, convert_to_default_units, ureg
from .algorithm_class import Algorithm

RETURN_KEYS = [
    "edge_impurity_concentration",
]


def run_calc_edge_impurity_concentration(
    edge_impurity_species: Impurity,
    q_parallel: Unitfull,
    SOL_power_loss_fraction: Unitfull,
    target_electron_temp: Unitfull,
    upstream_electron_temp: Unitfull,
    upstream_electron_density: Unitfull,
    kappa_e0: Unitfull,
    lengyel_overestimation_factor: Unitfull,
    reference_electron_density: Unitfull = 1.0 * ureg.n20,
    reference_ne_tau: Unitfull = 1.0 * ureg.n20 * ureg.ms,
) -> dict[str, Unitfull]:
    """Calculate the impurity concentration required to cool the scrape-off-layer using the Lengyel model.

    Args:
        edge_impurity_species: :term:`glossary link<edge_impurity_species>`
        reference_electron_density: :term:`glossary link<reference_electron_density>`
        reference_ne_tau: :term:`glossary link<reference_ne_tau>`
        q_parallel: :term:`glossary link<q_parallel>`
        SOL_power_loss_fraction: :term:`glossary link<SOL_power_loss_fraction>`
        target_electron_temp: :term:`glossary link<target_electron_temp>`
        upstream_electron_temp: :term:`glossary link<upstream_electron_temp>`
        upstream_electron_density: :term:`glossary link<upstream_electron_density>`
        kappa_e0: :term:`glossary link<kappa_e0>`
        lengyel_overestimation_factor: :term:`glossary link<lengyel_overestimation_factor>`

    Returns:
        :term:`edge_impurity_concentration`
    """
    atomic_data = read_atomic_data()

    L_int_integrator = build_L_int_integrator(
        atomic_data=atomic_data,
        impurity_species=edge_impurity_species,
        reference_electron_density=reference_electron_density,
        reference_ne_tau=reference_ne_tau,
    )

    edge_impurity_concentration = calc_required_edge_impurity_concentration(
        L_int_integrator=L_int_integrator,
        q_parallel=q_parallel,
        SOL_power_loss_fraction=SOL_power_loss_fraction,
        target_electron_temp=target_electron_temp,
        upstream_electron_temp=upstream_electron_temp,
        upstream_electron_density=upstream_electron_density,
        kappa_e0=kappa_e0,
        lengyel_overestimation_factor=lengyel_overestimation_factor,
    )

    local_vars = locals()
    return {key: convert_to_default_units(local_vars[key], key) for key in RETURN_KEYS}


calc_edge_impurity_concentration = Algorithm(
    function=run_calc_edge_impurity_concentration,
    return_keys=RETURN_KEYS,
)
