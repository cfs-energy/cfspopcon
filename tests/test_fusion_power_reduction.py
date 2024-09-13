import numpy as np

from cfspopcon.formulas.fusion_power.fusion_power_reduction import require_P_fusion_less_than_P_fusion_limit
from cfspopcon.unit_handling import magnitude_in_units, ureg


def test_require_P_fusion_less_than_P_fusion_limit():
    # Test it does not change the heavier_fuel_species_fraction if P_fusion is less than the limit
    P_fusion_upper_limit = 100.0 * ureg.MW
    P_fusion = 50.0 * ureg.MW
    heavier_fuel_species_fraction = 0.5

    new_heavier_fuel_species_fraction = require_P_fusion_less_than_P_fusion_limit(
        P_fusion_upper_limit=P_fusion_upper_limit,
        P_fusion=P_fusion,
        heavier_fuel_species_fraction=heavier_fuel_species_fraction,
    )
    np.testing.assert_allclose(magnitude_in_units(new_heavier_fuel_species_fraction, ureg.dimensionless), 0.5, rtol=1e-3)

    # Test it changes the heavier_fuel_species_fraction correctly if P_fusion is greater than the limit
    P_fusion_upper_limit = 100.0 * ureg.MW
    P_fusion = 104.16 * ureg.MW
    heavier_fuel_species_fraction = 0.5
    new_heavier_fuel_species_fraction = require_P_fusion_less_than_P_fusion_limit(
        P_fusion_upper_limit=P_fusion_upper_limit,
        P_fusion=P_fusion,
        heavier_fuel_species_fraction=heavier_fuel_species_fraction,
    )

    np.testing.assert_allclose(magnitude_in_units(new_heavier_fuel_species_fraction, ureg.dimensionless), 0.4, rtol=1e-2)
