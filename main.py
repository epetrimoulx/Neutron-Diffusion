import numpy as np

from Diffusion import diffusion_3d
from Graphing import *
from Material import Shape
from Integrate import rk4

"""
NOTES:
- The Grid mesh we have made in class creates boundary conditions relative to the grid size. The length, width, and height of the objects needs to be 
    dependant on the grid size.

- Learn how to make the grids better. Implement 3D grid. Try to make the size of the grid vary based on how big the objects are.
"""

BOX_LENGTH: int = 10
DIFFUSION_CONST: float = 0.1

def set_boundary_conditions(boundary_array: np.ndarray) -> np.ndarray:
    boundary_array[:, :, -1] = True
    boundary_array[:, :,  0] = True
    
    return boundary_array

def set_initial_conditions(init_array: np.ndarray, shape) -> np.ndarray:
    # init_array[:, :, -1] = 0.001 #shape.calc_num_neutrons() * shape.height
    # init_array[:, :,  0] = 0.001 #shape.calc_num_neutrons() * shape.height

    init_array.fill(0.01)
    
    return init_array

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

    k_effective =  total_density[1:] / total_density[:-1]
    return k_effective

def determine_k(k) -> None:
    if k > 1:
        print(f'The reaction has gone super-critical!')
    elif k == 1:
        print(f'The reaction has gone critical!')
    else:
        print(f'The reaction is sub-critical!')


def main():
    # Create Fuel Rod Objects
    fuel_rod_1 = Shape(40, 40, 40, 'Cube', 235, 'Uranium', 92)
    fuel_rod_2 = Shape(40, 40, 40, 'Cube', 235, 'Uranium', 92)

    print(fuel_rod_1)

    # Set up initial conditions for both Fuel Rod Objects
    init_condition_1 = np.zeros((
        fuel_rod_1.length,
        fuel_rod_1.width,
        fuel_rod_1.height
    ))

    init_condition_2 = np.zeros((
        fuel_rod_2.length,
        fuel_rod_2.width,
        fuel_rod_2.height
    ))


    # Set up boundary conditions for both Fuel Rod Objects
    boundary_1 = np.full(
        (
        fuel_rod_1.length,
        fuel_rod_1.width,
        fuel_rod_1.height
        ),False, dtype = bool
    )

    # boundary_2 = np.full(
    #     (
    #         BOX_LENGTH - fuel_rod_2.length,
    #         BOX_LENGTH - fuel_rod_2.width,
    #         BOX_LENGTH - fuel_rod_2.height
    #     ), False, dtype=bool
    # )

    set_boundary_conditions(boundary_1)
    # set_boundary_conditions(boundary_2)

    set_initial_conditions(init_condition_1, fuel_rod_1)
    # set_initial_conditions(init_condition_2, fuel_rod_2)

    # Initialize grid-spacing, timesteps, number of timesteps and the total time
    grid_spacing: float = 1e4 / (fuel_rod_1.length * fuel_rod_1.width * fuel_rod_1.height) # 1e4 added for memory management
    timestep = (grid_spacing ** 2 / 4 / DIFFUSION_CONST) * 0.25
    t_final = 10
    num_timesteps = int(t_final / timestep)

    result: np.ndarray = diffusion_3d(init_condition_1, grid_spacing, timestep, num_timesteps, boundary_1, diffusion_const=DIFFUSION_CONST)

    # Check for nan's
    if np.isnan(result).any():
        print(f'The reaction went super critical and an overflow occurred. Stopping the simulation.')
        exit()

    # Sum over x, y, z axis to get the density at different time steps
    total_density = np.sum(result, axis = (0, 1, 2))
    print(total_density[1], total_density[-1])
    time = np.linspace(0, t_final, num_timesteps)

    plot_neutron_density(total_density, time)

    k = growth_rate(total_density)

    plot_k_vs_time(k, time[:-1])

    plt.imshow(result[:, 25, :, 540], cmap='coolwarm', origin='lower')
    plt.colorbar(fraction=0.02)
    plt.show()




if __name__ == '__main__':
    main()