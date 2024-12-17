import numpy as np
from ipywidgets import interact, IntSlider
import ipywidgets as widgets

from Diffusion import *
from Graphing import *
from Material import Shape

BOX_LENGTH: int = 42 # [m]
DIFFUSION_CONST: float = 2.34e5 # [1/s]
NEUTRON_ENERGY: float = 2 # [MeV]
MeV_to_J: float = 1.6022e-13 # [MeV/J]
URANIUM_DENSITY: float = 18710.0 # [kg/m^3]
PLUTONIUM_DENSITY: float = 15600.0 # [kg/m^3]
AU_to_Kg: float = 1.6605e-27 # [au/kg]


def growth_rate(total_density: np.ndarray) -> np.ndarray:
    """
    Determines the growth rate of neutrons in the diffusion process. If the Neutron Multiplication factor (k) is
        equal to 1, The reaction is critical. If k is less than 1 the reaction is Sub-critical, and if k is greater than 1 the
        reaction is Supercritical

    :param total_density:
    :type: np.ndarray

    :return: average neutron multiplication factor
    :rtype np.ndarray

    :author: Evan Petrimoulx
    :date: November 5th 2024
    """
    k_effective = np.zeros(len(total_density))
    for i in range(1, len(total_density) - 1):
        k_effective[i+1] =  total_density[i+1] / total_density[i]

    return k_effective

def determine_k(k) -> None:
    if k > 1:
        print(f'The reaction has gone super-critical!')
    elif k == 1:
        print(f'The reaction has gone critical!')
    else:
        print(f'The reaction is sub-critical!')

def place_fuel_rods_in_grid(grid: np.ndarray[float], rod1_initial_condition: np.ndarray[float], rod2_initial_condition: np.ndarray[float ], separation: float) -> np.ndarray[float]:
    """
    Takes the initial conditions set up from the individual fuel rods and places them into the simulation grid at a
    specified separation from one another

    :param grid: The Simulation grid
    :type grid: np.ndarray[float]
    :param rod1_initial_condition: The first fuel rod initial conditions (initial neutron density)
    :type rod1_initial_condition: np.ndarray
    :type rod2_initial_condition: np.ndarray
    :param rod2_initial_condition: The second fuel rod initial conditions (initial neutron density)
    :param separation: The separation between the fuel rods and the initial conditions
    :type separation: float
    """

    # Get the size of the larger and smaller arrays
    larger_size = grid.shape
    smaller_size1 = rod1_initial_condition.shape
    smaller_size2 = rod2_initial_condition.shape

    # Calculate the center of the larger array
    center_larger_x, center_larger_y, center_larger_z = larger_size[0] // 2, larger_size[1] // 2, larger_size[2] // 2

    # Calculate the starting indices for placing the smaller first array
    start_x = center_larger_x - smaller_size1[0] // 2 + separation //2
    start_y = center_larger_y - smaller_size1[1] // 2
    start_z = center_larger_z - smaller_size1[2] // 2

    # Place the smaller array inside the larger one
    grid[start_x:start_x + smaller_size1[0], start_y:start_y + smaller_size1[1], start_z:start_z + smaller_size1[2]] = rod1_initial_condition

    # Calculate the starting indices for placing the second smaller array
    start_x = center_larger_x - smaller_size2[0] // 2 - separation//2
    start_y = center_larger_y - smaller_size2[1] // 2
    start_z = center_larger_z - smaller_size2[2] // 2

    grid[start_x:start_x + smaller_size2[0], start_y:start_y + smaller_size2[1],start_z:start_z + smaller_size2[2]] = rod2_initial_condition

    return grid

def get_energy(density: np.ndarray[float]) -> None:
    energy_init = density[0] * NEUTRON_ENERGY * MeV_to_J
    energy_final = density[-1] * NEUTRON_ENERGY * MeV_to_J
    print(f'\nThe energy in the system initially is {energy_init:.3e} J')
    print(f'The energy in the system at the end of the simulation is {energy_final:.3e} J\n')
    print(f'The energy emitted/lost was {energy_final - energy_init:.2e} J')


def main():
    # Create Fuel Rod Objects
    fuel_rod_1 = Shape(8, 8, 8, 'Cube', 235, 'Uranium', 92, URANIUM_DENSITY)
    fuel_rod_2 = Shape(8, 8, 8, 'Cube', 235, 'Uranium', 92, URANIUM_DENSITY)

    print(f'{fuel_rod_1} \n')
    print(f'{fuel_rod_2} \n')

    # Set up initial conditions for both Fuel Rod Objects
    initial_condition_1 = fuel_rod_1.set_initial_conditions()
    initial_condition_2 = fuel_rod_2.set_initial_conditions()

    # Initialize grid-spacing, timesteps, number of timesteps, total time, and density
    grid_spacing: float       = 1 / BOX_LENGTH
    timestep_size: float      = (grid_spacing ** 2 / 6 / DIFFUSION_CONST) * 0.25
    t_final: float            = 5e-8
    num_timesteps: int        = int(t_final / timestep_size)
    total_density: np.ndarray = np.zeros(num_timesteps)
    time: np.ndarray          = np.linspace(0, t_final, num_timesteps)

    # Create Grid
    grid = np.zeros((BOX_LENGTH, BOX_LENGTH, BOX_LENGTH))
    object_separation = 20

    # Embed fuel rods into the grid
    grid = place_fuel_rods_in_grid(grid, initial_condition_1, initial_condition_2, object_separation)

    # Set up boundary conditions
    boundary_grid = np.full((BOX_LENGTH, BOX_LENGTH, BOX_LENGTH), False)
    boundary_grid[-1, :, :] = True
    boundary_grid[:, -1, :] = True
    boundary_grid[0, :, :] = True
    boundary_grid[:, 0, :] = True


    # Plot the fuel rod starting positions
    plot_fuel_rod_positions(grid)

    # Diffuse
    result = diffusion_3d(grid, grid_spacing, timestep_size, num_timesteps, boundary_grid)

    # Check for nan's
    if np.isnan(result).any():
        print(f'The reaction went super critical and an overflow occurred. Stopping the simulation.')
        exit()
        
    # Calculate density
    for i in range(0, num_timesteps):
        total_density[i] = np.sum(result[:, :, :, i]) / result.size

    plt.figure()
    plt.title(f'Total Neutron Density over time')
    plt.xlabel('Time')
    plt.ylabel('Average Neutron Density')
    plt.plot(time, total_density)
    plt.show()

    # Create the slider widget for time step
    time_slider = IntSlider(
        value=0,  # Initial value
        min=0,  # Minimum time step
        max=result.shape[3] - 1,  # Maximum time step
        step=1,  # Increment step
        description='Time Step'
    )

    y_slider = IntSlider(
        value= result.shape[1] // 2,
        min=0,
        max= result.shape[1] - 1,
        step = 1,
        description='Y-axis'
    )

    # Use the interact function to link the slider to the plot function
    interact(plot_diffusion, time_step=time_slider, y_index=y_slider)

if __name__ == '__main__':
    main()