"""Module defining the AtomicData class, used for interfacing with radas files."""

import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np
import xarray as xr

from ...algorithm_class import Algorithm
from ...helpers import get_item
from ...named_options import AtomicSpecies
from ...unit_handling import Quantity, magnitude_in_units, ureg
from .coeff_interpolator import CoeffInterpolator


class AtomicData:
    """A class to manage atomic data for various species, providing facilities for accessing datasets directly or by constructing interpolators for different datasets (i.e. radiated power curves).

    Attributes:
    - atomic_data_directory (Path): Path to the directory containing atomic data files.
    - datasets (dict): A dictionary storing the atomic data for different species.
    - available_species (list): A list of species for which atomic data is available.
    - coronal_Lz_interpolators (dict): A dictionary of interpolators for coronal Lz values.
    - coronal_Z_interpolators (dict): A dictionary of interpolators for coronal Z values.
    - noncoronal_Lz_interpolators (dict): A dictionary of interpolators for non-coronal Lz values.
    - noncoronal_Z_interpolators (dict): A dictionary of interpolators for non-coronal Z values.
    """

    def __init__(self, atomic_data_directory: Path = Path() / "radas_dir") -> None:
        """Initializes the AtomicData object by loading atomic data from the specified directory.

        Parameters:
        - atomic_data_directory (Path): The path to the directory containing atomic data files.
        """
        self.atomic_data_directory = atomic_data_directory
        self.datasets = self.read_atomic_data(atomic_data_directory)  # Load atomic data into the datasets attribute
        self.available_species = list(self.datasets.keys())  # List available species based on the loaded datasets

        # Initialize dictionaries to hold interpolators for different data types and conditions
        self.coronal_Lz_interpolators: dict[AtomicSpecies, CoeffInterpolator] = dict()
        self.coronal_Z_interpolators: dict[AtomicSpecies, CoeffInterpolator] = dict()
        self.noncoronal_Lz_interpolators: dict[tuple[AtomicSpecies, float], CoeffInterpolator] = dict()
        self.noncoronal_Z_interpolators: dict[tuple[AtomicSpecies, float], CoeffInterpolator] = dict()

        self.species_ne_tau: dict[AtomicSpecies, xr.DataArray] = dict()
        self.ne_tau_units = ureg.m**-3 * ureg.s
        self.radas_git_hash: str = ""

        for species in self.available_species:
            dataset = self[species]

            ref = dict(
                reference_electron_density=dataset.reference_electron_density,
                reference_electron_temp=dataset.reference_electron_temp,
            )

            self.coronal_Lz_interpolators[species] = CoeffInterpolator(dataset.coronal_Lz, **ref)
            self.coronal_Z_interpolators[species] = CoeffInterpolator(dataset.coronal_mean_charge_state, **ref)

            self.species_ne_tau[species] = dataset["ne_tau"].pint.to(self.ne_tau_units).pint.dequantify()

            for ne_tau, dataset_at_single_ne_tau in dataset.groupby("dim_ne_tau"):
                subds = dataset_at_single_ne_tau.squeeze(dim="dim_ne_tau")
                self.noncoronal_Lz_interpolators[(species, ne_tau)] = CoeffInterpolator(subds.equilibrium_Lz, **ref)
                self.noncoronal_Z_interpolators[(species, ne_tau)] = CoeffInterpolator(subds.equilibrium_mean_charge_state, **ref)

            self._check_radas_git_hash(dataset.git_hash)

    def _check_radas_git_hash(self, test_git_hash: str) -> None:
        """Check that all of the datasets have the same git hash."""
        if self.radas_git_hash == "":
            self.radas_git_hash = test_git_hash
        elif self.radas_git_hash != test_git_hash:
            warnings.warn(
                f"Found multiple radas git hashes ({self.radas_git_hash} != {test_git_hash}) in the requested atomic data. Will set radas_git_hash = UNDEFINED.",
                stacklevel=2,
            )

    @staticmethod
    def read_atomic_data(atomic_data_directory: Path = Path() / "radas_dir") -> dict[AtomicSpecies, xr.Dataset]:
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
        # Ensure the atomic data directory and its 'output' subdirectory exist
        if not atomic_data_directory.exists():
            raise FileNotFoundError(f"atomic_data_directory ({atomic_data_directory.absolute()}) does not exist.")

        if not (atomic_data_directory / "output").exists():
            raise FileNotFoundError(
                f"atomic_data_directory ({atomic_data_directory}) does not contain a subfolder called 'output'. Make sure you have executed `poetry run radas` before calling this function."
            )

        atomic_data = dict()
        # Iterate through each netCDF file in the 'output' directory
        for file in (atomic_data_directory / "output").iterdir():
            if file.suffix == ".nc":  # Check if the file is a netCDF file
                species = file.stem  # Extract the species name from the file name
                try:
                    # Attempt to map the file name to an AtomicSpecies enum
                    species_enum = AtomicSpecies[species.capitalize()]
                    # Read the netCDF file into an xarray Dataset and quantify it with pint
                    atomic_data[species_enum] = xr.open_dataset(file).pint.quantify()
                except KeyError:
                    # If no matching AtomicSpecies enum is found, print a warning
                    print(f"No AtomicSpecies found corresponding to {species}")

        return atomic_data

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
        self, species: str | AtomicSpecies, ne_tau: float | Quantity, ne_tau_rel_tolerance: Optional[float | Quantity] = None
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
        self, species: str | AtomicSpecies, ne_tau: float | Quantity, ne_tau_rel_tolerance: Optional[float | Quantity] = None
    ) -> CoeffInterpolator:
        """Returns a noncoronal_Lz_interpolator for the specified species and ne_tau value."""
        ne_tau = self._get_nearest_ne_tau(species, ne_tau, ne_tau_rel_tolerance)

        return self.noncoronal_Lz_interpolators[(self.key_to_enum(species), ne_tau)]

    def get_noncoronal_Z_interpolator(
        self, species: str | AtomicSpecies, ne_tau: float | Quantity, ne_tau_rel_tolerance: Optional[float | Quantity] = None
    ) -> CoeffInterpolator:
        """Returns a noncoronal_Z_interpolator for the specified species and ne_tau value."""
        ne_tau = self._get_nearest_ne_tau(species, ne_tau, ne_tau_rel_tolerance)

        return self.noncoronal_Z_interpolators[(self.key_to_enum(species), ne_tau)]


@Algorithm.register_algorithm(return_keys=["atomic_data", "radas_git_hash"])
def read_atomic_data(radas_dir: Path) -> tuple[AtomicData, str]:
    """Construct an AtomicData interface using the atomic data in the specified directory."""
    atomic_data = AtomicData(get_item(radas_dir))
    return atomic_data, atomic_data.radas_git_hash
