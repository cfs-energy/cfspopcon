"""Module defining the AtomicData class, used for interfacing with radas files."""

import warnings
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar, Union, cast

import numpy as np
import xarray as xr

from ...algorithm_class import Algorithm
from ...helpers import get_item
from ...named_options import AtomicSpecies
from ...unit_handling import Quantity, magnitude_in_units, ureg
from .coeff_interpolator import CoeffInterpolator

K = TypeVar("K")
V = TypeVar("V")


class _LazyValueDict(dict[K, V]):
    """Dictionary that memoizes values loaded on first access.

    AtomicData uses this for datasets, ``ne_tau`` grids, and interpolators so
    object construction only indexes the RADAS files. The heavier xarray loads
    and interpolator construction stay deferred until a specific species or
    charge-state table is actually requested.
    """

    def __init__(self, loader: Callable[[K], V]) -> None:
        super().__init__()
        self._loader = loader

    def __missing__(self, key: K) -> V:
        """Populate and cache a missing value via the configured loader."""
        value = self._loader(key)
        self[key] = value
        return value


class AtomicData:
    """Manage RADAS datasets and interpolators for the available species.

    The object records which netCDF files are present up front, but it opens
    each dataset and builds each interpolator lazily. That keeps startup cheap
    for workflows that only touch a small subset of species or ``ne_tau``
    slices while preserving the same public access patterns as the eager
    implementation.
    """

    def __init__(self, atomic_data_directory: Path = Path() / "radas_dir") -> None:
        """Index the RADAS output directory and initialize the lazy caches.

        Args:
            atomic_data_directory: Path to the directory containing RADAS
                output files.
        """
        self.atomic_data_directory = atomic_data_directory
        # Keep an index of files up front, then load each dataset/interpolator
        # only when it is requested by a downstream formula.
        self.atomic_data_files = self.find_atomic_data_files(atomic_data_directory)
        self.datasets: dict[AtomicSpecies, xr.Dataset] = _LazyValueDict(self._load_dataset)
        self.available_species = list(self.atomic_data_files.keys())

        # Build interpolators only when a particular species/table is requested.
        self.coronal_Lz_interpolators: dict[AtomicSpecies, CoeffInterpolator] = _LazyValueDict(self._load_coronal_Lz_interpolator)
        self.coronal_Z_interpolators: dict[AtomicSpecies, CoeffInterpolator] = _LazyValueDict(self._load_coronal_Z_interpolator)
        self.noncoronal_Lz_interpolators: dict[tuple[AtomicSpecies, float], CoeffInterpolator] = _LazyValueDict(
            self._load_noncoronal_Lz_interpolator
        )
        self.noncoronal_Z_interpolators: dict[tuple[AtomicSpecies, float], CoeffInterpolator] = _LazyValueDict(
            self._load_noncoronal_Z_interpolator
        )

        self.species_ne_tau: dict[AtomicSpecies, xr.DataArray] = _LazyValueDict(self._load_species_ne_tau)
        self.ne_tau_units = ureg.m**-3 * ureg.s

        self._radas_version: str = ""
        self._radas_version_checked_species: set[AtomicSpecies] = set()

    @property
    def radas_version(self) -> str:
        """Return the aggregate RADAS version string after checking all datasets.

        RADAS stores the version metadata per netCDF file, so this property
        forces each available species dataset through the lazy loader before it
        exposes the aggregate version summary.
        """
        for species in self.available_species:
            _ = self[species]
        return self._radas_version

    def _load_dataset(self, species: AtomicSpecies) -> xr.Dataset:
        """Open, quantify, and version-check the dataset for one species."""
        dataset = cast("xr.Dataset", xr.open_dataset(self.atomic_data_files[species]).pint.quantify())
        if species not in self._radas_version_checked_species:
            self._check_radas_version(getattr(dataset, "radas_version", "UNDEFINED"))
            self._radas_version_checked_species.add(species)
        return dataset

    def _get_reference_values(self, species: AtomicSpecies) -> dict[str, Quantity]:
        """Return the shared normalization references used by interpolators."""
        dataset = self[species]
        return dict(
            reference_electron_density=dataset.reference_electron_density,
            reference_electron_temp=dataset.reference_electron_temp,
        )

    def _load_coronal_Lz_interpolator(self, species: AtomicSpecies) -> CoeffInterpolator:
        """Build the coronal radiated-power interpolator for one species."""
        dataset = self[species]
        return CoeffInterpolator(dataset.coronal_Lz, **self._get_reference_values(species))

    def _load_coronal_Z_interpolator(self, species: AtomicSpecies) -> CoeffInterpolator:
        """Build the coronal mean-charge-state interpolator for one species."""
        dataset = self[species]
        return CoeffInterpolator(dataset.coronal_mean_charge_state, **self._get_reference_values(species))

    def _load_species_ne_tau(self, species: AtomicSpecies) -> xr.DataArray:
        """Extract the supported ``ne_tau`` grid for one species."""
        return cast("xr.DataArray", self[species]["ne_tau"].pint.to(self.ne_tau_units).pint.dequantify())

    def _load_noncoronal_Lz_interpolator(self, key: tuple[AtomicSpecies, float]) -> CoeffInterpolator:
        """Build the non-coronal radiated-power interpolator for one species and ``ne_tau``."""
        species, ne_tau = key
        dataset = self[species].sel(dim_ne_tau=ne_tau)
        # ``sel`` can keep a length-1 dimension depending on xarray indexing;
        # strip it so the interpolator always sees a pure 2D coefficient table.
        if "dim_ne_tau" in dataset.dims:
            dataset = dataset.squeeze(dim="dim_ne_tau")
        return CoeffInterpolator(dataset.equilibrium_Lz, **self._get_reference_values(species))

    def _load_noncoronal_Z_interpolator(self, key: tuple[AtomicSpecies, float]) -> CoeffInterpolator:
        """Build the non-coronal mean-charge-state interpolator for one species and ``ne_tau``."""
        species, ne_tau = key
        dataset = self[species].sel(dim_ne_tau=ne_tau)
        if "dim_ne_tau" in dataset.dims:
            dataset = dataset.squeeze(dim="dim_ne_tau")
        return CoeffInterpolator(dataset.equilibrium_mean_charge_state, **self._get_reference_values(species))

    def _check_radas_version(self, test_version: str) -> None:
        """Fold one dataset's RADAS version into the aggregate version summary.

        The first observed version becomes the summary value. Later mismatches
        raise a warning so callers know the summary is no longer uniform across
        the loaded files.
        """
        if self._radas_version == "":
            self._radas_version = test_version
        elif self._radas_version != test_version:
            warnings.warn(
                f"Found multiple RADAS versions ({self._radas_version} != {test_version}) in the requested atomic data. Retaining the first observed version.",
                stacklevel=2,
            )

    @staticmethod
    def find_atomic_data_files(atomic_data_directory: Path = Path() / "radas_dir") -> dict[AtomicSpecies, Path]:
        """Return the available RADAS netCDF files indexed by species.

        This helper only scans the directory structure; it does not open any of
        the datasets.
        """
        if not atomic_data_directory.exists():
            raise FileNotFoundError(f"atomic_data_directory ({atomic_data_directory.absolute()}) does not exist.")

        if not (atomic_data_directory / "output").exists():
            raise FileNotFoundError(
                f"atomic_data_directory ({atomic_data_directory}) does not contain a subfolder called 'output'. Make sure you have executed `poetry run radas` before calling this function."
            )

        atomic_data_files: dict[AtomicSpecies, Path] = dict()
        for file in (atomic_data_directory / "output").iterdir():
            if file.suffix == ".nc":
                species = file.stem
                try:
                    atomic_data_files[AtomicSpecies[species.capitalize()]] = file
                except KeyError:
                    print(f"No AtomicSpecies found corresponding to {species}")

        return atomic_data_files

    @classmethod
    def read_atomic_data(cls, atomic_data_directory: Path = Path() / "radas_dir") -> dict[AtomicSpecies, xr.Dataset]:
        """Reads atomic data from netCDF files located in the specified directory.

        This function scans a directory for netCDF files (.nc), each representing atomic data for a different species.
        It expects these files to be in a subdirectory named 'output'. The data is read into an xarray Dataset, quantified
        with pint for unit handling, and stored in a dictionary with keys corresponding to AtomicSpecies enums.

        Parameters:
        - atomic_data_directory (Path): The path to the directory containing the 'output' folder with netCDF files.

        Returns:
        - dict: A dictionary where keys are AtomicSpecies enums and values are xarray Datasets of the atomic data.

        Raises:
        - FileNotFoundError: If the atomic_data_directory or its 'output' subdirectory does not exist.
        """
        return {
            species: xr.open_dataset(file).pint.quantify() for species, file in cls.find_atomic_data_files(atomic_data_directory).items()
        }

    @staticmethod
    def key_to_enum(species: Union[str, AtomicSpecies]) -> AtomicSpecies:
        """Converts a species identifier to an AtomicSpecies enum.

        This method allows for flexible specification of species, accepting either a string (which is then capitalized
        and matched to an AtomicSpecies enum) or an AtomicSpecies enum directly.

        Parameters:
        - species (Union[str, AtomicSpecies]): The species identifier, either a string name or an AtomicSpecies enum.

        Returns:
        - AtomicSpecies: The corresponding AtomicSpecies enum.
        """
        if isinstance(species, str):
            species = AtomicSpecies[species.capitalize()]
        return species

    def __getitem__(self, species: Union[str, AtomicSpecies]) -> xr.Dataset:
        """Allows direct access to the atomic data dataset for a given species using dictionary-style indexing.

        Parameters:
        - species (Union[str, AtomicSpecies]): The species identifier, either as a string or an AtomicSpecies enum.

        Returns:
        - The xarray Dataset corresponding to the atomic data of the requested species.
        """
        return self.datasets[self.key_to_enum(species)]

    def get_coronal_Lz_interpolator(self, species: str | AtomicSpecies) -> CoeffInterpolator:
        """Returns a coronal_Lz_interpolator for the specified species and ne_tau value."""
        return self.coronal_Lz_interpolators[self.key_to_enum(species)]

    def get_coronal_Z_interpolator(self, species: str | AtomicSpecies) -> CoeffInterpolator:
        """Returns a coronal_Z_interpolator for the specified species and ne_tau value."""
        return self.coronal_Z_interpolators[self.key_to_enum(species)]

    def _get_nearest_ne_tau(
        self, species: str | AtomicSpecies, ne_tau: float | Quantity, ne_tau_rel_tolerance: float | Quantity | None = None
    ) -> float:
        """Find the nearest ne_tau value to the requested ne_tau value.

        If a tolerance is specified, raise an error if the nearest ne_tau value is outside the specified tolerance.
        If a tolerance is not specified, raise a warning if the nearest ne_tau value is outside the specified tolerance.
        """
        if not isinstance(ne_tau, float):
            ne_tau = float(magnitude_in_units(ne_tau, self.ne_tau_units))

        nearest_ne_tau = float(self.species_ne_tau[self.key_to_enum(species)].sel(dim_ne_tau=ne_tau, method="nearest"))

        tolerance_specified = ne_tau_rel_tolerance is not None
        default_ne_tau_rel_tolerance = 1e-2

        if tolerance_specified and not isinstance(ne_tau_rel_tolerance, float):
            ne_tau_rel_tolerance = float(magnitude_in_units(ne_tau_rel_tolerance, self.ne_tau_units))  # type: ignore [arg-type]
        elif not tolerance_specified:
            ne_tau_rel_tolerance = default_ne_tau_rel_tolerance

        if np.abs((nearest_ne_tau - ne_tau) / ne_tau) > ne_tau_rel_tolerance:
            message = f"Requested ne_tau = {ne_tau}. Nearest value {nearest_ne_tau} was further than {ne_tau_rel_tolerance} * {ne_tau} from the requested value."

            if tolerance_specified:
                raise KeyError(message)
            else:
                warnings.warn(message, stacklevel=2)

        return nearest_ne_tau

    def get_noncoronal_Lz_interpolator(
        self, species: str | AtomicSpecies, ne_tau: float | Quantity, ne_tau_rel_tolerance: float | Quantity | None = None
    ) -> CoeffInterpolator:
        """Returns a noncoronal_Lz_interpolator for the specified species and ne_tau value."""
        ne_tau = self._get_nearest_ne_tau(species, ne_tau, ne_tau_rel_tolerance)

        return self.noncoronal_Lz_interpolators[(self.key_to_enum(species), ne_tau)]

    def get_noncoronal_Z_interpolator(
        self, species: str | AtomicSpecies, ne_tau: float | Quantity, ne_tau_rel_tolerance: float | Quantity | None = None
    ) -> CoeffInterpolator:
        """Returns a noncoronal_Z_interpolator for the specified species and ne_tau value."""
        ne_tau = self._get_nearest_ne_tau(species, ne_tau, ne_tau_rel_tolerance)

        return self.noncoronal_Z_interpolators[(self.key_to_enum(species), ne_tau)]


@Algorithm.register_algorithm(return_keys=["atomic_data", "radas_version"])
def read_atomic_data(radas_dir: Path) -> tuple[AtomicData, str]:
    """Construct an AtomicData interface and report the checked RADAS version summary."""
    atomic_data = AtomicData(get_item(radas_dir))
    return atomic_data, atomic_data.radas_version
