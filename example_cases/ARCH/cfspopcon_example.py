import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cfspopcon
from cfspopcon.unit_handling import ureg
import warnings
import numpy as np
import os
from pathlib import Path
warnings.simplefilter(action='ignore', category=FutureWarning)

def get_profiles(i, j):
    # Get CFSPOPCON Profile
    average_electron_density = dataset['average_electron_density'].values[i]
    electron_density_peaking = dataset['electron_density_peaking'].values[i][j]
    dilution = dataset['dilution'].values[i][j]
    ion_density_peaking = dataset['ion_density_peaking'].values[i][j]
    average_electron_temp = dataset['average_electron_temp'].values[j]
    temperature_peaking = dataset['temperature_peaking'].values
    average_ion_temp = dataset['average_ion_temp'].values[j]
    electron_density_profile = (
        average_electron_density * electron_density_peaking * ((1.0 - rho**2.0) ** (electron_density_peaking - 1.0))
    )
    fuel_ion_density_profile = (
        average_electron_density * dilution * (ion_density_peaking) * ((1.0 - rho**2.0) ** (ion_density_peaking - 1.0))
    )
    electron_temp_profile = average_electron_temp * temperature_peaking * ((1.0 - rho**2.0) ** (temperature_peaking - 1.0))
    ion_temp_profile = average_ion_temp * temperature_peaking * ((1.0 - rho**2.0) ** (temperature_peaking - 1.0))

    # Get Frank Profile
    density_sep = 1.00
    temperature_sep = 0.1
    density_0 = 4.00 # This is an estimate, can't actually find this exact value in Frank but maybe in another paper?
    temperature_0 = 24.9
    alpha_density = 1.1
    alpha_temperature = 1.5
    
    density_profile = density_sep + (density_0 - density_sep) * (1 - rho**2) ** alpha_density
    temperature_profile = temperature_sep + (temperature_0 - temperature_sep) * (1 - rho**2) ** alpha_temperature
    
    return {'cfspopcon': {'density': electron_density_profile, 'temperature': electron_temp_profile},
            'frank': {'density': density_profile, 'temperature': temperature_profile}}


# Find temperature and density array indices in CFSPOPCON that produce density profiles most similar to the ones in Frank.
# Also find temperature and density array indices in CFSPOPCON that produce temperature profiles most similar to the ones in Frank. 
def find_matching_cfspopcon_profile():
    min_density_diff = 99999
    min_temp_diff = 99999
    indices = {'Dens Profile': {'Dens Index': 0, 'Temp Index': 0}, 
               'Temp Profile': {'Dens Index': 0, 'Temp Index': 0}}
    profiles = None
    for i in range(40):
        for j in range(30):
            profiles = get_profiles(i, j)
            new_density_diff = np.linalg.norm(profiles['cfspopcon']['density'] - profiles['frank']['density'])
            if new_density_diff < min_density_diff:
                min_density_diff = new_density_diff
                indices['Dens Profile']['Dens Index'] = i
                indices['Dens Profile']['Temp Index'] = j
            new_temp_diff = np.linalg.norm(profiles['cfspopcon']['temperature'] - profiles['frank']['temperature'])
            if new_temp_diff < min_temp_diff:
                min_temp_diff = new_temp_diff
                indices['Temp Profile']['Dens Index'] = i
                indices['Temp Profile']['Temp Index'] = j
    return indices

# Plot CFSPOPCON Density, Frank Density, CFSPOPCON Temperature, Frank Temperature Profiles
def plot_profiles():
    indices = find_matching_cfspopcon_profile()
    dpdi = indices['Dens Profile']['Dens Index']
    dpti = indices['Dens Profile']['Temp Index']
    tpdi = indices['Temp Profile']['Dens Index']
    tpti = indices['Temp Profile']['Temp Index']

    fig, axs = plt.subplots(2, 1)
    plt.subplots_adjust(hspace=0.4)
    axs[0].plot(rho, get_profiles(dpdi, dpti)['cfspopcon']['density'], color='red', label='cfspopcon')
    axs[0].plot(rho, get_profiles(dpdi, dpti)['frank']['density'], color='blue', label='frank')
    axs[0].set_title(f'Density Profiles - density={dataset["average_electron_density"].values[dpdi]}, temp={dataset["average_electron_temp"].values[dpti]}')
    axs[0].legend()
    axs[1].plot(rho, get_profiles(tpdi, tpti)['cfspopcon']['temperature'], color='red', label='cfspopcon')
    axs[1].plot(rho, get_profiles(tpdi, tpti)['frank']['temperature'], color='blue', label='frank')
    axs[1].set_title(f'Temperature Profiles - density={dataset["average_electron_density"].values[tpdi]}, temp index={dataset["average_electron_temp"].values[tpti]}')
    axs[1].legend()
    plt.show()

# Make a popcon
def plot_popcon():
    plot_style = cfspopcon.read_plot_style(os.path.join(os.getcwd(), "example_cases/ARCH/plot_popcon_arch.yaml"))
    cfspopcon.plotting.make_plot(dataset, plot_style, points=points, title="POPCON example", output_dir=Path(os.path.join(os.getcwd(), "example_cases/ARCH")))
    plt.show()

if __name__ == '__main__':
    np.set_printoptions(threshold=np.Inf)

    npoints = 50
    rho = np.linspace(0, 1, num=npoints, endpoint=False)
    
    # Load input paramters from yaml file
    input_parameters, algorithm, points = cfspopcon.read_case(os.path.join(os.getcwd(), "example_cases/ARCH/input_arch.yaml"))
    algorithm.validate_inputs(input_parameters)
    dataset = xr.Dataset(input_parameters)
    algorithm.update_dataset(dataset, in_place=True)

    plot_profiles()
    plot_popcon()