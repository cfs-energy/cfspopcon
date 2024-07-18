"""Read in information about various energy confinement scalings from the energy_confinement_scalings.yaml file."""

from __future__ import annotations

from importlib.resources import as_file, files
from typing import Any, ClassVar

import yaml

from ...algorithm_class import Algorithm


class ConfinementScaling:
    """Class to handle different energy confinement scalings."""

    instances: ClassVar[dict[str, ConfinementScaling]] = dict()

    def __init__(self, name: str, data: dict[str, Any]) -> None:
        """Initialises an energy confinement scaling from a block of the energy_confinement_scalings.yaml file."""
        self.instances[name] = self

        self.data = data

        self.name: str = name
        self.reference: str = data["metadata"]["reference"]
        self.notes: str = data["metadata"]["notes"]
        self.regime: str = data["metadata"]["regime"]

        self.constant = data["params"]["constant"]
        self.mass_ratio_alpha = data["params"]["mass_ratio_alpha"]
        self.field_on_axis_alpha = data["params"]["field_on_axis_alpha"]
        self.plasma_current_alpha = data["params"]["plasma_current_alpha"]
        self.input_power_alpha = data["params"]["input_power_alpha"]
        self.major_radius_alpha = data["params"]["major_radius_alpha"]
        self.triangularity_alpha = data["params"]["triangularity_alpha"]
        self.inverse_aspect_ratio_alpha = data["params"]["inverse_aspect_ratio_alpha"]
        self.areal_elongation_alpha = data["params"]["areal_elongation_alpha"]
        self.separatrix_elongation_alpha = data["params"]["separatrix_elongation_alpha"]
        self.average_density_alpha = data["params"]["average_density_alpha"]
        self.qstar_alpha = data["params"]["qstar_alpha"]


@Algorithm.register_algorithm(return_keys=[])
def read_confinement_scalings() -> None:
    """Reads the energy confinement scalings from an energy_confinement_scalings.yaml file."""
    with as_file(files("cfspopcon.formulas.energy_confinement").joinpath("energy_confinement_scalings.yaml")) as filepath:
        with open(filepath) as f:
            data = yaml.safe_load(f)

    for scaling_name, scaling_data in data.items():
        ConfinementScaling(scaling_name, scaling_data)
