from matplotlib.animation import FuncAnimation

from Diffusion import *
from Graphing import *
from Material import Shape

BOX_LENGTH: int = 40
DIFFUSION_CONST: float = 0.1
INITIAL_CONDITION: float = 1.0
COLLISION_SPEED: float = 0.1 # How fast the fuel rods are being hit together

def set_initial_conditions(init_array: np.ndarray) -> np.ndarray:
    init_array.fill(INITIAL_CONDITION)

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

def place_fuel_rods_in_grid(grid, boundary_grid, rod, position='left'):
    """
    Add the neutron density of a rod into the simulation grid based on its position
    :param grid: The simulation grid
    :param boundary_grid: The boundary grid
    :param rod: The rod
    :param position: The starting position of the rod
    :return: Grid values with the rods inside as well as a matching boundary condition grid
    """

    if position == 'left':
        # Place the rod on the left side of the grid
        x_start = 0
        x_end = min(grid.shape[0], int(rod.length))  # Limit x_end to the grid size
    elif position == 'right':
        # Place the rod on the right side of the grid
        x_start = max(0, grid.shape[0] - int(rod.length))  # Start from the far right
        x_end = grid.shape[0]
    else:
        print(f'Position {position} is not valid. Placing object in the center\n')
        x_start = max(0, int(rod.x_center - rod.length // 2))
        x_end = min(grid.shape[1], int(rod.x_center + rod.length // 2))

    # Rods should start at the same height (Head on collisions only for now)
    y_start = max(0, int(rod.y_center - rod.length // 2))
    y_end = min(grid.shape[1], int(rod.y_center + rod.length // 2))
    z_start = max(0, int(rod.z_center - rod.length // 2))
    z_end = min(grid.shape[2], int(rod.z_center + rod.length // 2))

    grid[x_start:x_end, y_start:y_end, z_start:z_end] += INITIAL_CONDITION
    boundary_grid[x_start:x_end, y_start:y_end, z_start:z_end] = True

    return grid, boundary_grid


def main():
    # Create Fuel Rod Objects
    fuel_rod_1 = Shape(4, 4, 4, 'Cube', 235, 'Uranium', 92, (5, BOX_LENGTH // 2, BOX_LENGTH // 2))
    fuel_rod_2 = Shape(4, 4, 4, 'Cube', 235, 'Uranium', 92, (35, BOX_LENGTH // 2, BOX_LENGTH // 2))

    print(f'{fuel_rod_1} \n')
    print(f'{fuel_rod_2} \n')

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

    set_initial_conditions(init_condition_1)
    set_initial_conditions(init_condition_2)

    # Initialize grid-spacing, timesteps, number of timesteps, total time, and density
    grid_spacing: float       = 1e4 / (BOX_LENGTH**3) # 1e4 added for memory management
    timestep_size: float      = (grid_spacing ** 2 / 4 / DIFFUSION_CONST) * 0.25
    t_final: int              = 10
    num_timesteps: int        = int(t_final / timestep_size)
    total_density: np.ndarray = np.zeros(num_timesteps)
    result: np.ndarray        = np.zeros((num_timesteps, BOX_LENGTH, BOX_LENGTH, BOX_LENGTH))
    time: np.ndarray          = np.linspace(0, t_final, num_timesteps)

    # At the start of the simulation, the rods are far apart and havent yet collided
    has_collided: bool = False

    # Create Grid
    grid = np.zeros((BOX_LENGTH, BOX_LENGTH, BOX_LENGTH))
    boundary_grid = np.full((BOX_LENGTH, BOX_LENGTH, BOX_LENGTH), False)

    # Embed fuel rods into the grid
    grid, boundary_grid = place_fuel_rods_in_grid(grid, boundary_grid, fuel_rod_1, position='left')
    grid, boundary_grid = place_fuel_rods_in_grid(grid, boundary_grid, fuel_rod_2, position='right')


    # Info for first step in diffusion process
    nx, ny, nz = grid.shape
    diffusion: np.ndarray = np.zeros((nx, ny, nz, num_timesteps), dtype=np.float64)
    d = DIFFUSION_CONST * timestep_size / grid_spacing ** 3
    diffusion[:, :, :, 0] = np.copy(grid)
    total_density[0] = np.sum(diffusion[:, :, :, 0])

    for timestep in range(1, num_timesteps):

        # Move the rods closer together
        if (not has_collided) and ((timestep * COLLISION_SPEED) - round(timestep * COLLISION_SPEED) < 1e-9):
            boundary_grid, has_collided = smash_fuel_rods_together(boundary_grid, has_collided)
            plot_fuel_rod_positions(boundary_grid)

        # Diffuse
        diffusion = diffusion_3d(nx, ny, nz, diffusion, d, timestep_size, timestep, boundary_grid)
        total_density[timestep] = np.sum(diffusion[:, :, :, timestep])

        # Check for nan's
        if np.isnan(result).any():
            print(f'The reaction went super critical and an overflow occurred. Stopping the simulation.')
            exit()

    result = diffusion

    plot_neutron_density_evolution(result, time)
    k = growth_rate(total_density)

    for i in range(len(total_density) - 1):
        determine_k(k[i])

    plot_k_vs_time(k, time[:-1])

    plt.imshow(np.transpose(result[:, -1//2, :, -1//2]), cmap='coolwarm', origin='lower')
    plt.colorbar(fraction=0.02)
    plt.show()


if __name__ == '__main__':
    main()