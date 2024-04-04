from __future__ import annotations
import yaml 
from importlib.resources import files, as_file
from typing import Any, ClassVar

class ConfinementScaling:
    instances: ClassVar[dict[str, ConfinementScaling]] = dict()

    def __init__(self, name: str, data: dict[str, Any]) -> None:
        """Initialises an energy confinement scaling from a block of the energy_confinement_scalings.yaml file."""
        self.instances[name] = self

        self.name: str = name
        self.documentation: str = data["metadata"]["documentation"]
        self.notes: str = data["metadata"]["notes"]
        self.regime: str = data["metadata"]["regime"]

        self.constant = data["params"].get("constant", 0.0)
        self.mass_ratio_alpha = data["params"].get("mass_ratio_alpha", 0.0)
        self.field_on_axis_alpha = data["params"].get("field_on_axis_alpha", 0.0)
        self.plasma_current_alpha = data["params"].get("plasma_current_alpha", 0.0)
        self.input_power_alpha = data["params"].get("input_power_alpha", 0.0)
        self.major_radius_alpha = data["params"].get("major_radius_alpha", 0.0)
        self.triangularity_alpha = data["params"].get("triangularity_alpha", 0.0)
        self.inverse_aspect_ratio_alpha = data["params"].get("inverse_aspect_ratio_alpha", 0.0)
        self.areal_elongation_alpha = data["params"].get("areal_elongation_alpha", 0.0)
        self.separatrix_elongation_alpha = data["params"].get("separatrix_elongation_alpha", 0.0)
        self.average_density_alpha = data["params"].get("average_density_alpha", 0.0)
        self.qstar_alpha = data["params"].get("qstar_alpha", 0.0)
    
def read_confinement_scalings() -> dict[str, Any]:
    """Reads the energy confinement scalings from an energy_confinement_scalings.yaml file."""
    with as_file(files("cfspopcon.new_formulas.energy_confinement").joinpath("energy_confinement_scalings.yaml")) as filepath:
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)
    
    for scaling_name, scaling_data in data.items():
        ConfinementScaling(scaling_name, scaling_data)
    
    return data

if __name__=="__main__":
    read_confinement_scalings()