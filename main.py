import numpy as np
from numpy import ndarray

from Diffusion import diffusion_3d
from Material import Shape
from matplotlib import pyplot as plt

"""
NOTES:
- The Grid mesh we have made in class creates boundary conditions relative to the grid size. The length, width, and height of the objects needs to be 
    dependant on the grid size.

- Learn how to make the grids better. Implement 3D grid. Try to make the size of the grid vary based on how big the objects are.
"""

NUM_PARTICLES: int = 30
BOX_LENGTH: int = 10
DIFFUSION_CONST: float = 1.0

def set_boundary_conditions(boundary_array: np.ndarray) -> np.ndarray:
    boundary_array[:, :, -1] = True
    return boundary_array

def set_initial_conditions(init_array: np.ndarray) -> np.ndarray:
    init_array[:, :, -1] = +1.0
    return init_array


def main():
    # Create Fuel Rod Objects
    fuel_rod_1 = Shape(10, 10, 10, 'Cube', 235, 'Uranium', 92)
    fuel_rod_2 = Shape(10, 10, 10, 'Cube', 235, 'Uranium', 92)

    print(fuel_rod_1)

    # Set up initial conditions for both Fuel Rod Objects
    init_condition_1 = np.zeros((fuel_rod_1.length, fuel_rod_1.width, fuel_rod_1.height))
    init_condition_2 = np.zeros((BOX_LENGTH - fuel_rod_1.length, BOX_LENGTH + fuel_rod_1.width, BOX_LENGTH + fuel_rod_1.height))

    # Set up boundary conditions for both Fuel Rod Objects
    boundary_1 = np.full((fuel_rod_1.length, fuel_rod_1.width, fuel_rod_1.height),False, dtype = bool)
    boundary_2 = np.full((BOX_LENGTH - fuel_rod_1.length, BOX_LENGTH + fuel_rod_1.width, BOX_LENGTH + fuel_rod_1.height), False, dtype=bool)

    set_boundary_conditions(boundary_1)
    set_boundary_conditions(boundary_2)

    set_initial_conditions(init_condition_1)
    set_initial_conditions(init_condition_2)

    # Initialize grid-spacing, timesteps, number of timesteps and the total time
    grid_spacing: float = 1.0 / NUM_PARTICLES
    timestep = (grid_spacing ** 2 / 4 / DIFFUSION_CONST) * 0.25
    t_final = 0.25
    num_timesteps = int(t_final / timestep)

    result: np.ndarray = diffusion_3d(init_condition_1, grid_spacing, timestep, num_timesteps, boundary_1, diffusion_const=DIFFUSION_CONST)
    
    x, y, z, time = np.split(result, indices_or_sections=4, axis=-1)
    print(result.shape)

if __name__ == '__main__':
    main()