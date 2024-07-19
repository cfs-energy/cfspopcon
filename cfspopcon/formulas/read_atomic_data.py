"""Module defining the AtomicData class, used for interfacing with radas files."""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator  # type: ignore[import-untyped]

from ..algorithm_class import Algorithm
from ..named_options import AtomicSpecies
from ..unit_handling import magnitude


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

    # Define constants for selecting interpolator types
    CoronalLz = 1
    CoronalZ = 2
    NoncoronalLz = 3
    NoncoronalZ = 4

    def __init__(self, atomic_data_directory: Path = Path() / "radas_dir") -> None:
        """Initializes the AtomicData object by loading atomic data from the specified directory.

        Parameters:
        - atomic_data_directory (Path): The path to the directory containing atomic data files.
        """
        self.atomic_data_directory = atomic_data_directory
        self.datasets = self.read_atomic_data(atomic_data_directory)  # Load atomic data into the datasets attribute
        self.available_species = list(self.datasets.keys())  # List available species based on the loaded datasets

        # Initialize dictionaries to hold interpolators for different data types and conditions
        self.coronal_Lz_interpolators: dict[AtomicSpecies, RegularGridInterpolator] = dict()
        self.coronal_Z_interpolators: dict[AtomicSpecies, RegularGridInterpolator] = dict()
        self.noncoronal_Lz_interpolators: dict[tuple[AtomicSpecies, float], RegularGridInterpolator] = dict()
        self.noncoronal_Z_interpolators: dict[tuple[AtomicSpecies, float], RegularGridInterpolator] = dict()
        self.grid_limits: dict[AtomicSpecies, tuple[float, float, float, float]] = dict()

        for species in self.available_species:
            dataset = self.get_dataset(species)
            self.coronal_Lz_interpolators[species] = self.build_interpolator(dataset.coronal_Lz)
            self.coronal_Z_interpolators[species] = self.build_interpolator(dataset.coronal_mean_charge_state)

            max_temp = dataset.dim_electron_temp.max().item()
            min_temp = dataset.dim_electron_temp.min().item()
            max_density = dataset.dim_electron_density.max().item()
            min_density = dataset.dim_electron_density.min().item()
            self.grid_limits[species] = (max_temp, min_temp, max_density, min_density)

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
        return self.get_dataset(species)

    def get_dataset(self, species: Union[str, AtomicSpecies]) -> xr.Dataset:
        """Retrieves the atomic data dataset for a specified species.

        Parameters:
        - species (Union[str, AtomicSpecies]): The species identifier, either as a string or an AtomicSpecies enum.

        Returns:
        - The xarray Dataset corresponding to the atomic data of the requested species.
        """
        return self.datasets[self.key_to_enum(species)]

    @staticmethod
    def build_interpolator(z_values: xr.DataArray) -> RegularGridInterpolator:
        """Builds a bivariate spline interpolator for the provided dataset.

        This method creates an interpolator based on logarithmic transformations of the temperature, density,
        and z_values data, facilitating interpolation in log space for better accuracy over wide ranges.

        Parameters:
        - z_values (xr.DataArray): The xarray DataArray containing the data to interpolate, indexed by electron temperature and density.

        Returns:
        - RegularGridInterpolator: The bivariate spline interpolator object.
        """
        tiny = np.finfo(np.float64).tiny

        def log10_with_floor(x: Union[xr.DataArray, np.ndarray, float]) -> Union[xr.DataArray, np.ndarray, float]:
            floored_log: float = np.log10(np.maximum(x, tiny))
            return floored_log

        # Remove rows or columns with all negative or zero values
        z_values = z_values.where(~np.all(z_values <= 0.0, axis=0), drop=True)
        z_values = z_values.where(~np.all(z_values <= 0.0, axis=1), drop=True)

        # Logarithmic transformation of the temperature, density, and z_values for interpolation
        return RegularGridInterpolator(
            points=(
                log10_with_floor(z_values.dim_electron_temp),
                log10_with_floor(z_values.dim_electron_density),
            ),
            values=log10_with_floor(
                magnitude(z_values.transpose("dim_electron_temp", "dim_electron_density")).to_numpy()  # type:ignore[union-attr]
            ),
            method="cubic",
            bounds_error=True,
        )

    def get_interpolator(self, kind: int, species: Union[str, AtomicSpecies], ne_tau: float = np.inf) -> RegularGridInterpolator:
        """Retrieves or creates a bivariate spline interpolator for a given species and physical condition.

        This method manages a cache of interpolator objects to avoid redundant computations. It also
        validates the requested conditions against the available data.

        Parameters:
        - kind (int): The type of data to interpolate, specified by class constants (e.g., CoronalLz, NoncoronalZ).
        - species (Union[str, AtomicSpecies]): The species identifier, either as a string or an AtomicSpecies enum.
        - ne_tau (float): The electron density times ionization time product, for non-coronal conditions. Default is infinity, indicating coronal equilibrium.

        Returns:
        - RegularGridInterpolator: The requested interpolator object.

        Raises:
        - FileNotFoundError: If no dataset is available for the requested species.
        - RuntimeError: For invalid ne_tau values or misapplication of ne_tau in coronal conditions.
        """
        species = self.key_to_enum(species)  # Convert species to enum if necessary
        if species not in self.available_species:
            raise FileNotFoundError(
                f"Requested data for {species.name} but no corresponding {species.name.lower()}.nc dataset is available in {(self.atomic_data_directory / 'output').absolute()}"
            )

        noncoronal_key = (species, ne_tau)  # Create a unique noncoronal_key for caching
        dataset = self.get_dataset(species)  # Retrieve the dataset for the species

        # Validate ne_tau and manage exceptions for coronal conditions
        if (ne_tau < np.inf) and (ne_tau not in dataset.dim_ne_tau):
            raise RuntimeError(
                f"Requested a value of ne_tau ({ne_tau} m^-3 s) which was not in the available values ({list(dataset.dim_ne_tau.values)}). Check the documentation on configuring radas."
            )
        if (kind in [self.CoronalLz, self.CoronalZ]) and (ne_tau < np.inf):
            raise RuntimeError("Requested coronal data. ne_tau will have no effect.")

        # Create or retrieve the appropriate interpolator based on the kind and conditions
        if kind == self.CoronalLz:
            if species not in self.coronal_Lz_interpolators:
                self.coronal_Lz_interpolators[species] = self.build_interpolator(dataset.coronal_Lz)
            return self.coronal_Lz_interpolators[species]

        if kind == self.CoronalZ:
            if species not in self.coronal_Z_interpolators:
                self.coronal_Z_interpolators[species] = self.build_interpolator(dataset.coronal_mean_charge_state)
            return self.coronal_Z_interpolators[species]

        if kind == self.NoncoronalLz:
            if noncoronal_key not in self.noncoronal_Lz_interpolators:
                self.noncoronal_Lz_interpolators[noncoronal_key] = self.build_interpolator(dataset.equilibrium_Lz.sel(dim_ne_tau=ne_tau))
            return self.noncoronal_Lz_interpolators[noncoronal_key]

        if kind == self.NoncoronalZ:
            if noncoronal_key not in self.noncoronal_Z_interpolators:
                self.noncoronal_Z_interpolators[noncoronal_key] = self.build_interpolator(
                    dataset.equilibrium_mean_charge_state.sel(dim_ne_tau=ne_tau)
                )
            return self.noncoronal_Z_interpolators[noncoronal_key]

    def eval_interpolator(
        self,
        electron_density: Union[xr.DataArray, np.ndarray, float],
        electron_temp: Union[xr.DataArray, np.ndarray, float],
        kind: int,
        species: Union[str, AtomicSpecies],
        ne_tau: float = np.inf,
        allow_extrapolation: bool = False,
        grid: bool = True,
        coords: Optional[dict[str, Union[xr.DataArray, np.ndarray, float]]] = None,
    ) -> xr.DataArray:
        """Evaluates the interpolator for given electron densities and temperatures, returning interpolated values.

        N.b. Not recommended for performant code! It is better to directly work with the interpolators, and use this
        only as a convenience function.

        This method allows for the interpolation of data (e.g., ionization rates) over a grid of electron densities
        and temperatures for a specific species and physical condition.

        If allow_extrapolation, off-grid points are replaced by their nearest on-grid neighbours.

        Parameters:
        - electron_density (xr.DataArray): The electron densities for which to interpolate data.
        - electron_temp (xr.DataArray): The electron temperatures for which to interpolate data.
        - kind (int): The type of data to interpolate, specified by class constants.
        - species (Union[str, AtomicSpecies]): The species identifier, either as a string or an AtomicSpecies enum.
        - ne_tau (float): The electron density times ionization time product, for non-coronal conditions. Default is infinity, indicating coronal equilibrium.
        - allow_extrapolation (bool): Whether to allow extrapolation beyond the data range. Default is False.
        - grid (bool): Whether the interpolation is done on a meshgrid of electron densities and temperatures. Default is True.

        Returns:
        - xr.DataArray: The interpolated values as an xarray DataArray.

        Raises:
        - AssertionError: If the input ranges for temperature or density are beyond the available data range, unless extrapolation is allowed.
        """
        species = self.key_to_enum(species)
        interpolator = self.get_interpolator(kind=kind, species=species, ne_tau=ne_tau)  # Retrieve the appropriate interpolator

        if coords is None:
            coords = dict(
                dim_electron_density=electron_density,
                dim_electron_temp=electron_temp,
            )  # Prepare coordinates for the result DataArray

        # Handle optional extrapolation
        if allow_extrapolation:
            # Adjust electron_temp and electron_density to fit within the dataset's bounds, if necessary
            electron_temp, electron_density = self.nearest_neighbour_off_grid(
                species=species, electron_temp=electron_temp, electron_density=electron_density
            )
        else:
            # Assert that the electron_temp and electron_density are within the dataset's bounds
            self.assert_on_grid(species=species, electron_temp=electron_temp, electron_density=electron_density)

        # Perform the interpolation and convert back from logarithmic values
        if grid:
            electron_temp, electron_density = np.meshgrid(electron_temp, electron_density)
        interpolated_values = np.power(10, interpolator((np.log10(electron_temp), np.log10(electron_density))))

        return xr.DataArray(interpolated_values, coords=coords)  # Return the interpolated values as an xarray DataArray

    def nearest_neighbour_off_grid(
        self,
        species: AtomicSpecies,
        electron_temp: Union[xr.DataArray, np.ndarray, float],
        electron_density: Union[xr.DataArray, np.ndarray, float],
    ) -> tuple[Union[xr.DataArray, np.ndarray, float], Union[xr.DataArray, np.ndarray, float]]:
        """Replaces off-grid points with their nearest on-grid neighbour."""
        max_temp, min_temp, max_density, min_density = self.grid_limits[species]
        electron_temp = np.minimum(electron_temp, max_temp)
        electron_temp = np.maximum(electron_temp, min_temp)
        electron_density = np.minimum(electron_density, max_density)
        electron_density = np.maximum(electron_density, min_density)
        return electron_temp, electron_density

    def assert_on_grid(
        self,
        species: AtomicSpecies,
        electron_temp: Union[xr.DataArray, np.ndarray, float],
        electron_density: Union[xr.DataArray, np.ndarray, float],
    ) -> None:
        """Raises an AssertionError if any points are off-grid."""
        max_temp, min_temp, max_density, min_density = self.grid_limits[species]
        assert np.max(electron_temp) <= max_temp, f"{np.max(electron_temp)} > {max_temp}"
        assert np.min(electron_temp) >= min_temp, f"{np.min(electron_temp)} < {min_temp}"
        assert np.max(electron_density) <= max_density, f"{np.max(electron_density)} > {max_density}"
        assert np.min(electron_density) >= min_density, f"{np.min(electron_density)} < {min_density}"


@Algorithm.register_algorithm(return_keys=["atomic_data"])
def read_atomic_data(radas_dir: Path) -> AtomicData:
    """Construct an AtomicData interface using the atomic data in the specified directory."""
    if isinstance(radas_dir, xr.DataArray):
        return AtomicData(radas_dir.item())
    else:
        return AtomicData(radas_dir)
