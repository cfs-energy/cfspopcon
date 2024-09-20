"""Routine to calculate the upstream SOL collisionality."""



from ....unit_handling import Unitfull, ureg, wraps_ufunc


@wraps_ufunc(
    return_units=dict(upstream_SOL_collisionality=ureg.dimensionless),
    input_units=dict(separatrix_electron_density=ureg.m**-3, separatrix_electron_temp=ureg.eV, parallel_connection_length=ureg.m),
)
def calc_upstream_SOL_collisionality(
    separatrix_electron_density: Unitfull,
    separatrix_electron_temp: Unitfull,
    parallel_connection_length: Unitfull,
) -> Unitfull:
    """Calculate the upstream SOL collisionality.

    Equation XX

    Args:
        separatrix_electron_density: [m^-3]
        separatrix_electron_temp: [eV]
        parallel_connection_length: [m]

    Returns:
        upstream_SOL_collisionality
    """
    return 1e-16 * parallel_connection_length * separatrix_electron_density / separatrix_electron_temp**2
