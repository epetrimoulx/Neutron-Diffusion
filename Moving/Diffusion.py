"""
File to deal with the Diffusion Equation

The diffusion constant is equal to µ = λ_t v_neut/3, where λt represents neutron transport free
path, and v_neut represents neutron speed. The neutron rate of formation constant is
equal to η = v_neut(v - 1)/λf , where v represents secondary neutrons created by fission
(the -1 accounts for the neutron causing fission being consumed in the reaction), and
λf represents neutron fission free path. Using U235 values from Table (7) we obtain:
µ = 2.345E+05 (m2/s) and η = 1.896E+08 (1/s).
"""

import numba
import numpy as np

@numba.njit
def diffusion_3d(
        nx: int,
        ny: int,
        nz: int,
        diffusion: np.ndarray,
        d: float,
        timestep: float,
        curr_time_index: int,
        boundary: np.ndarray,
        velocity: np.ndarray,
        grid_spacing: float,
        dirichlet_values: np.ndarray = None,
        boundary_type: str = 'neumann',
        reaction_rate: float = 1.001 #1.896e8
) -> np.ndarray:
    """
    Simulates the 3D diffusion-reaction process over a grid with Dirichlet or Neumann boundary conditions at a single timestep with moving neutron sources

    This function evolves the neutron density `n` over time according to the diffusion-reaction equation:

        ∂n/∂t = D ∇²n + \eta n - v・∇n

    where `D` is the diffusion constant, and `\eta` is the reaction rate constant.

    :param nx: size of the grid in x
    :type nx: int
    :param ny: size of the grid in y
    :type ny: int
    :param nz: size of the grid in z
    :type nz: int
    :param diffusion: The diffusion variable
    :type diffusion: np.ndarray
    :param d: The diffusion constant multiplied by the timestep and the spacial step
    :type d: float
    :param curr_time_index: The current timestep this function is being called at to evaluate
    :type curr_time_index: int
    :param boundary: Boolean 3D array marking boundary (`True`) and interior (`False`) points.
    :type boundary: numpy.ndarray
    :param velocity: 4D array containing the velocity at all points in the grid. Gives the "flow" towards the center of the grid. Velocity should match the speed of the moving boundary conditions.
    :type velocity: np.ndarray
    :param grid_spacing: The spacing between grid points.
    :type grid_spacing: float
    :param timestep: Time increment for each simulation step.
    :type timestep: float
    :param dirichlet_values: Fixed concentration values at boundary points (for Dirichlet conditions).
                             If `None`, Neumann (no-flux) conditions are applied.
    :type dirichlet_values: numpy.ndarray, optional
    :param boundary_type: Type of boundary conditions.
    :type boundary_type: str
    :param reaction_rate: Reaction rate constant `η`. Default is 1.896e8.
    :type reaction_rate: float, optional

    :return: 4D array of neutron density values over time with shape `(x, y, z, time)`.
    :rtype: numpy.ndarray

    :behavior:
      - Applies periodic boundaries with modular indexing.
      - Dirichlet boundary conditions are used if `dirichlet_values` is provided;
        otherwise, Neumann (no-flux) conditions are applied, keeping boundary values constant.

    :author: Evan Petrimoulx
    :date: November 4th 2024
    """

    for ix in range(0, nx):
        for iy in range(0, ny):
            for iz in range(0, nz):
                if not boundary[ix, iy, iz]:
                    laplacian = (
                            diffusion[(ix + 1) % nx, iy, iz, curr_time_index - 1] +
                            diffusion[(ix - 1) % nx, iy, iz, curr_time_index - 1] +
                            diffusion[ix, (iy + 1) % ny, iz, curr_time_index - 1] +
                            diffusion[ix, (iy - 1) % ny, iz, curr_time_index - 1] +
                            diffusion[ix, iy, (iz + 1) % nz, curr_time_index - 1] +
                            diffusion[ix, iy, (iz - 1) % nz, curr_time_index - 1] -
                            6 * diffusion[ix, iy, iz, curr_time_index - 1]
                    )

                    # Compute advection term
                    vx, vy, vz = velocity[ix, iy, iz]
                    grad_x = (diffusion[(ix + 1) % nx, iy, iz, curr_time_index - 1] -
                              diffusion[(ix - 1) % nx, iy, iz, curr_time_index - 1]) / (2 * grid_spacing)
                    grad_y = (diffusion[ix, (iy + 1) % ny, iz, curr_time_index - 1] -
                              diffusion[ix, (iy - 1) % ny, iz, curr_time_index - 1]) / (2 * grid_spacing)
                    grad_z = (diffusion[ix, iy, (iz + 1) % nz, curr_time_index - 1] -
                              diffusion[ix, iy, (iz - 1) % nz, curr_time_index - 1]) / (2 * grid_spacing)

                    advection = vx * grad_x + vy * grad_y + vz * grad_z

                    # Calculate total diffusion for the timestep
                    diffusion[ix, iy, iz, curr_time_index] = (
                            diffusion[ix, iy, iz, curr_time_index - 1]
                            + timestep * (
                                    d * laplacian
                                    - advection
                                    + reaction_rate * diffusion[ix, iy, iz, curr_time_index - 1]
                            )
                    )
                else:
                    # Neumann Boundary Conditions
                    if boundary_type == 'neumann':
                        diffusion[ix, iy, iz, curr_time_index] = diffusion[ix, iy, iz, curr_time_index - 1]

                    # Dirichlet Boundary Conditions
                    elif boundary_type == 'dirichlet' and dirichlet_values is not None:
                        diffusion[ix, iy, iz, curr_time_index] = dirichlet_values[ix, iy, iz, curr_time_index - 1]

                    else:
                        # Robin Boundary Conditions Here
                        diffusion[ix, iy, iz, curr_time_index] = diffusion[ix, iy, iz, curr_time_index - 1]
    return diffusion

@numba.njit
def smash_fuel_rods_together(boundary_grid: np.ndarray[bool], has_collided: bool = False, distance_travelled: float = 1):
    """
        Takes the objects in the grid and moves them closer together until they collide.

        This is done by splitting the boundary grid array in half, finding all indices in the new grid where one of the objects is located,
        and moving it towards the center. The same is done for the second half of the array and the second object. This is done
        until both objects are touching the center of the array (or their respective edges in the split array). When this happens,
        they have met in the middle and are touching!

        :param boundary_grid: The boundary grid array
        :type boundary_grid: np.ndarray[bool]
        :param has_collided: Boolean flag to indicate if the objects have collided.
        :type has_collided: bool
        :param distance_travelled: Distance travelled in meters.
        :type distance_travelled: float

        :return: boundary grid with objects moved 1 time increment and status of collision
        :rtype: numpy.ndarray[float], bool

        :author: Evan Petrimoulx
        :date: November 4th 2024
        """

    # Get grid size
    x, y, z = boundary_grid.shape


    # Split grid in 2 along x
    left_boundary_grid: np.ndarray[bool] = boundary_grid[:x//2, :, :]
    right_boundary_grid: np.ndarray[bool] = boundary_grid[x//2:, :, :]

    # Get current position of objects via boundary condition locations
    left_object_location = np.where(left_boundary_grid == True)
    right_object_location = np.where(right_boundary_grid == True)


    # Create a copy of the array to store the updated values
    updated_left = np.copy(left_boundary_grid)
    updated_right = np.copy(right_boundary_grid)

    # Iterate over the True indices
    for x, y, z in zip(*left_object_location):
        if x + 1 < left_boundary_grid.shape[0]:  # Check if the incremented index is within bounds
            updated_left[x, y, z] = False
            updated_left[x + 1, y, z] = True

    for x, y, z in zip(*right_object_location):
        if x - 1 >= 0:
            updated_right[x, y, z] = False
            updated_right[x - 1, y, z] = True

    # Recombine into total boundary grid array
    boundary_grid = np.concatenate((updated_left, updated_right), axis=0)

    if (updated_left[-1, :, :].any() == True) and (updated_right[0, :, :].any() == True):
        has_collided = True

    return boundary_grid, has_collided










### TEST ###
def smash_fuel_rods_together_TEST(boundary_grid, has_collided, velocity_grid, grid_spacing, timestep_size):
    """
    Move the rods closer together based on their velocities.

    :param boundary_grid: Boolean grid indicating the locations of the rods
    :type boundary_grid: np.ndarray
    :param has_collided: Flag indicating whether the rods have collided
    :type has_collided: bool
    :param velocity_grid: Velocity grid for the simulation
    :type velocity_grid: np.ndarray
    :param grid_spacing: Spatial grid spacing
    :type grid_spacing: float
    :param timestep_size: Simulation timestep size
    :type timestep_size: float
    :return: Updated boundary grid and collision flag
    :rtype: (np.ndarray, bool)
    """
    # Calculate displacement (distance moved during this timestep)
    displacement = velocity_grid[:, :, :, 0] * timestep_size / grid_spacing  # x-direction only for now

    # Copy the current boundary grid to update it
    updated_boundary_grid = np.zeros_like(boundary_grid)

    # Move boundary points based on displacement
    for x in range(boundary_grid.shape[0]):
        for y in range(boundary_grid.shape[1]):
            for z in range(boundary_grid.shape[2]):
                if boundary_grid[x, y, z]:
                    # Calculate the new position of the boundary point
                    new_x = x + int(velocity_grid[x, y, z, 0] * timestep_size)
                    new_y = y + int(velocity_grid[x, y, z, 1] * timestep_size)
                    new_z = z + int(velocity_grid[x, y, z, 2] * timestep_size)

                    # Ensure new positions remain within the grid bounds
                    new_x = max(0, min(boundary_grid.shape[0] - 1, new_x))
                    new_y = max(0, min(boundary_grid.shape[1] - 1, new_y))
                    new_z = max(0, min(boundary_grid.shape[2] - 1, new_z))

                    updated_boundary_grid[new_x, new_y, new_z] = True

                    # Ensure new_x stays within bounds
                    if 0 <= new_x < boundary_grid.shape[0]:
                        updated_boundary_grid[new_x, y, z] = True

    # Check for collision: if the left and right rods have overlapping boundary points
    overlap = np.logical_and(updated_boundary_grid[:boundary_grid.shape[0] // 2],
                             updated_boundary_grid[boundary_grid.shape[0] // 2:])
    if np.any(overlap):
        has_collided = True
        print(f"The fuel rods have collided!")

    return updated_boundary_grid, has_collided