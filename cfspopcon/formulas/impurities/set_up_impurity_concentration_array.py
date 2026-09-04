"""Calculate the mean charge state of an impurity for given plasma conditions."""

from ...algorithm_class import Algorithm
from ...unit_handling import Unitfull


@Algorithm.register_algorithm(return_keys=["impurity_concentration"])
def set_up_impurity_concentration_array(intrinsic_impurity_concentration: Unitfull) -> Unitfull:
    """Set up the impurity concentration array, starting with the intrinsic impurities.

    Args:
        intrinsic_impurity_concentration: :term:`glossary link<intrinsic_impurity_concentration>`

    Returns:
        :term:`impurity_concentration`
    """
    impurity_concentration = intrinsic_impurity_concentration.copy()
    return impurity_concentration
