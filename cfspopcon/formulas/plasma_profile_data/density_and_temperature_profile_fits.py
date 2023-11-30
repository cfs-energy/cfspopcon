"""Realistic functional forms for T and n for POPCON analysis.

private communication, P. Rodriguez-Fernandez (MIT PSFC), 2020

Description:
    This functional form imposes:
        1) tanh pedestal for T and n.
        2) Linear aLT profile from 0 at rho=0 to X at rho=x_a,
            where X is the specified core aLT value (default 2.0)
            and x_a is calculated by matching specified temperature_peaking (peaking)
        3) Flat aLT profile from rho=x_a to rho=1-width_ped, where
            width_ped is the pedestal width (default 0.05).
        4) T and n share the same x_a, and aLn is calculated by matching
            specified nu_n (peaking)
        5) Pedestal is rescaled to match specified volume averages for
            T and n.

Notes:
    - Not all combinations of aLT and temperature_peaking are valid. If aLT is too low,
        temperature_peaking cannot be excessively high and viceversa. The code will not
        crash, but will give profiles that do not match the specified
        temperature    peaking.
        e.g. aLT = 2.0 requires temperature_peaking to be within [1.5,3.0]

    - It is not recommended to change width_ped from the default value,
        since the Look-Up-Table hard-coded was computed using
        width_ped=0.05

    - If rho-grid is passed as argument, it is recommended to have equally
        spaced 100 points.

Example use:

    T_avol = 7.6
    n_avol = 3.1
    temperature_peaking   = 2.5
    nu_n   = 1.4

    x, T, n = evaluate_density_and_temperature_profile_fits( T_avol, n_avol, temperature_peaking, nu_n, aLT = 2.0 )

    Optionally:
        - rho-grid can be passed (100 points recommended)
        - Pedestal width can be passed (0.05 recommended)

____________________________________________________________________
"""  # TODO: figure out valid regions of fits and print a warning when they are exceeded

import warnings
from functools import cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray
from scipy.interpolate import RectBivariateSpline  # type: ignore[import-untyped]

plasma_profiles_directory = Path(__file__).parent


def load_dataframe(dataset: str, df_name: str) -> pd.DataFrame:
    """Load specified dataframe for given dataset."""
    filepath = plasma_profiles_directory / dataset / f"{df_name}.csv"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = pd.read_csv(filepath, index_col=[0, 1], header=[0, 1])

    return df


@cache
def get_df_interpolator(dataset: str, df_name: str) -> RectBivariateSpline:
    """Return an interpolator for the given dataframe of the specified dataset."""
    df = load_dataframe(dataset, df_name)
    interpolator = RectBivariateSpline(
        [np.float64(x[1]) for x in df.columns.values],
        [np.float64(x[1]) for x in df.index.values],
        df.T.values,
    )
    return interpolator


def evaluate_density_and_temperature_profile_fits(
    T_avol: float,
    n_avol: float,
    temperature_peaking: float,
    nu_n: float,
    aLT: float = 2.0,
    width_ped: float = 0.05,
    rho: Optional[NDArray[np.float64]] = None,
    dataset: str = "PRF",
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:  # TODO: fill out docstring
    """Evaluate temperature-density profile fits."""
    # ---- Get interpolator functions corresponding to this dataset
    width_interpolator = get_df_interpolator(dataset=dataset, df_name="width")
    aLT_interpolator = get_df_interpolator(dataset=dataset, df_name="aLT")

    # ---- Find parameters consistent with peaking
    x_a = width_interpolator(aLT, temperature_peaking)[0]
    aLn = aLT_interpolator(x_a, nu_n)[0]

    # ---- Evaluate profiles
    x, T, _ = evaluate_profile(T_avol, width_ped=width_ped, aLT_core=aLT, width_axis=x_a, rho=rho)
    x, n, _ = evaluate_profile(n_avol, width_ped=width_ped, aLT_core=aLn, width_axis=x_a, rho=rho)

    return x, T, n


def evaluate_profile(
    Tavol: float,
    aLT_core: float,
    width_axis: float,
    width_ped: float = 0.05,
    rho: Optional[NDArray[np.float64]] = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    r"""This function generates a profile from :math:`\langle T \rangle`, aLT and :math:`x_a`.

    Example:
            x, T, temperature_peaking = evaluate_profile(7.6, 2.0, 0.2)
    """
    # ~~~~ Grid
    if rho is None:
        x = np.linspace(0, 1, 100)
    else:
        x = rho

    ix_c = np.argmin(np.abs(x - (1 - width_ped)))  # Extend of core
    ix_a = np.min([ix_c, np.argmin(np.abs(x - width_axis))])  # Extend of axis

    # ~~~~ aLT must be different from zero, adding non-rational small offset
    aLT_core = aLT_core + np.pi * 1e-8

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Functional Forms (normalized to pedestal temperature)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~

    # ~~~~ Pedestal
    # Notes:
    #         - Because width_ped and Teped in my function represent the top values, I need to rescale them
    #         - The tanh does not result exactly in the top value (since it's an asymptote), so I need to correct for it

    wped_tanh = width_ped / 1.5  # The pedestal width in the tanh formula is 50% inside the pedestal-top width
    Tedge_aux = 1 / 2 * (1 + np.tanh((1 - x - (wped_tanh / 2)) / (wped_tanh / 2)))
    Tedge = Tedge_aux[int(ix_c) :] / Tedge_aux[ix_c]

    # ~~~~ Core
    Tcore_aux = np.e ** (aLT_core * (1 - width_ped - x))
    Tcore = Tcore_aux[ix_a : int(ix_c)]

    # ~~~~ Axis
    Taxis_aux = np.e ** (aLT_core * (-1 / 2 * x**2 / width_axis - 1 / 2 * width_axis + 1 - width_ped))
    Taxis = Taxis_aux[:ix_a]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Analytical Integral ("pre-factor")
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Pedestal contribution (solved with Matematica)
    I1 = -0.0277778 * width_ped * (-23.3473 + 14.6132 * width_ped)

    # Core and axis contributions
    I23 = (
        1
        / aLT_core**2
        * (
            (width_axis * aLT_core * np.e ** (width_axis * aLT_core / 2) + 1) * np.e ** (-aLT_core * (width_axis + width_ped - 1))
            + aLT_core * width_ped
            - aLT_core
            - 1
        )
    )

    # Total (this is the factor that relates Teped to Tavol)
    I = 2 * (I1 + I23)  # noqa: E741

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Evaluation of the profile
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~

    Teped = Tavol / I
    T: NDArray[np.float64] = Teped * np.hstack((Taxis, Tcore, Tedge)).ravel()
    if np.isclose(Tavol, 0.0) and np.isclose(T[0], 0.0):
        peaking = 0.0
    else:
        peaking = float(T[0] / Tavol)

    return x, T, peaking


def load_metadata(dataset: str) -> dict[str, str]:
    r"""Load dataset metadata from YAML file.

    Args:
        dataset: name of subfolder that holds metadata.yaml

    Returns:
         metadata
    """
    filepath = plasma_profiles_directory / dataset / "metadata.yaml"
    with open(filepath) as f:
        metadata: dict[str, str] = yaml.safe_load(f)
    return metadata


def get_datasets() -> list[str]:
    """Get a list of names of valid datasets.

    Every immediate subdirectory of the source folder represents a dataset

    Returns:
         [str]*N, list of names of valid datasets
    """
    datasets = [f.name for f in plasma_profiles_directory.iterdir() if (f.is_dir() and not f.name.startswith("_"))]

    return datasets
