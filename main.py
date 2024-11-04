import numpy as np
from Diffusion import diffusion_3d
from Material import Shape
from matplotlib import pyplot as plt

"""
NOTES:
- The Grid mesh we have made in class creates boundary conditions relative to the grid size. The length, width, and height of the objects needs to be 
    dependant on the grid size.

- Learn how to make the grids better. Implement 3D grid. Try to make the size of the grid vary based on how big the objects are.

- Note: We assume there is no diffusion in the other directions of the fuel-rods, and all the neutrons are emmitted from one face. The surface is x, y, and the 
    diffusion direction is in Z.
"""

NUM_PARTICLES = 30
BOX_LENGTH = 10
DIFFUSION_CONST = 1.0

def set_boundary_conditions(boundary_array):
    boundary_array[:, :, -1] = True
    return boundary_array

def set_initial_conditions(init_array):
    init_array[:, :, -1] = +1.0
    return init_array

def main():
    # Create Fuel Rod Objects
    fuel_rod_1 = Shape(1, 1, 1, 'Cube', 235, 'Uranium', 92)
    fuel_rod_2 = Shape(1, 1, 1, 'Cube', 235, 'Uranium', 92)

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
    grid_spacing = 1.0 / NUM_PARTICLES
    timestep = (grid_spacing ** 2 / 4 / DIFFUSION_CONST) * 0.25
    t_final = 0.25
    num_timesteps = int(t_final / timestep)

    result = diffusion_3d(init_condition_1, boundary_1, DIFFUSION_CONST, grid_spacing, timestep, num_timesteps)
    
    x, y, z, time = np.split(result)

    print(result.shape)

if __name__ == '__main__':
    main()