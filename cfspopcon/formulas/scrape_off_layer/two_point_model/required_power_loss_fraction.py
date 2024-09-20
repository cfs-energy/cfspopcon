"""Calculate the SOL power loss fraction required to achieve a specified target electron temperature."""

from typing import Union

import numpy as np
import xarray as xr

from ....unit_handling import Unitfull


def calc_required_SOL_power_loss_fraction(
    target_electron_temp_basic: Unitfull,
    f_other_target_electron_temp: Union[float, xr.DataArray],
    SOL_momentum_loss_fraction: Unitfull,
    required_target_electron_temp: Unitfull,
) -> Union[float, xr.DataArray]:
    """Calculate the SOL radiated power fraction required to reach a desired target electron temperature.

    This equation is equation 15 of :cite:`stangeby_2018`, rearranged for $f_{cooling}$.

    Args:
        target_electron_temp_basic: from target_electron_temp module [eV]
        f_other_target_electron_temp: from target_electron_temp module [~]
        SOL_momentum_loss_fraction: fraction of momentum lost in SOL [~]
        required_target_electron_temp: what target temperature do we want? [eV]

    Returns:
        SOL_power_loss_fraction [~]
    """
    required_SOL_power_loss_fraction = xr.DataArray(
        1.0
        - np.sqrt(
            required_target_electron_temp
            / target_electron_temp_basic
            * (1.0 - SOL_momentum_loss_fraction) ** 2
            / f_other_target_electron_temp
        )
    )

    return required_SOL_power_loss_fraction.clip(min=0.0)
