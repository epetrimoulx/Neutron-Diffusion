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
        dirichlet_values: np.ndarray = None,
        boundary_type: str = 'neumann',
        reaction_rate: float = 1.2 #1.896e8
) -> np.ndarray:
    """
    Simulates the 3D diffusion-reaction process over a grid with Dirichlet or Neumann boundary conditions at a single timestep.

    This function evolves the neutron density `n` over time according to the diffusion-reaction equation:

        ∂n/∂t = D ∇²n + \eta n

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
                    diffusion[ix, iy, iz, curr_time_index] = (diffusion[ix, iy, iz, curr_time_index - 1] + 0.25 * d * (
                            diffusion[(ix + 1) % nx, iy, iz, curr_time_index - 1] +
                            diffusion[(ix - 1) % nx, iy, iz, curr_time_index - 1] +
                            diffusion[ix, (iy + 1) % ny, iz, curr_time_index - 1] +
                            diffusion[ix, (iy - 1) % ny, iz, curr_time_index - 1] +
                            diffusion[ix, iy, (iz + 1) % nz, curr_time_index -1] +
                            diffusion[ix, iy, (iz - 1) % nz, curr_time_index] - 6 * diffusion[ix, iy, iz, curr_time_index - 1]) +
                            reaction_rate * diffusion[ix, iy, iz, curr_time_index - 1] * timestep)
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

