"""Calculate the line averaged electron density."""

from ...algorithm_class import Algorithm
from ...unit_handling import Unitfull


@Algorithm.register_algorithm(return_keys=["line_averaged_electron_density"])
def calc_line_averaged_electron_density(average_electron_density: Unitfull, line_averaged_density_frac: float) -> Unitfull:
    """Calculate the line averaged electron density as a fix ratio of the average electron density.

    Args:
        average_electron_density: [1e19 m^-3] :term:`glossary link<average_electron_density>`
        line_averaged_density_frac: [~] :term:`glossary link<line_averaged_density_frac>`

    Returns:
        :term:`line_avearged_electron_density` [1e19 m^-3]
    """
    return line_averaged_density_frac * average_electron_density
