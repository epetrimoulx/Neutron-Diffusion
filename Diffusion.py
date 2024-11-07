"""
File to deal with the Diffusion Equation

The diffusion constant is equal to µ = λ_t v_neut/3, where λt represents neutron transport free
path, and v_neut represents neutron speed. The neutron rate of formation constant is
equal to η = v_neut(ν − 1)/λf , where ν represents secondary neutrons created by fission
(the −1 accounts for the neutron causing fission being consumed in the reaction), and
λf represents neutron fission free path. Using U235 values from Table (7) we obtain:
µ = 2.345E+05 (m2/s) and η = 1.896E+08 (1/s).
"""

import numba
import numpy as np

@numba.njit
def diffusion_3d(
        init_condition: np.ndarray,
        grid_spacing: float,
        timestep: float,
        num_timesteps: int,
        boundary: np.ndarray,
        dirichlet_values: np.ndarray = None,
        boundary_type: str = 'neumann',
        diffusion_const: float = 1.1, #2.34e5,
        reaction_rate: float = 1.2 #1.896e8
) -> np.ndarray:
    """
    Simulates the 3D diffusion-reaction process over a grid with Dirichlet or Neumann boundary conditions.

    This function evolves the neutron density `n` over time according to the diffusion-reaction equation:

        ∂n/∂t = D ∇²n + \eta n

    where `D` is the diffusion constant, and `\eta` is the reaction rate constant.

    :param init_condition: Initial neutron density array (3D ndarray).
    :type init_condition: numpy.ndarray
    :param boundary: Boolean 3D array marking boundary (`True`) and interior (`False`) points.
    :type boundary: numpy.ndarray
    :param grid_spacing: Distance between grid points in the spatial domain.
    :type grid_spacing: float
    :param timestep: Time increment for each simulation step.
    :type timestep: float
    :param num_timesteps: Number of timesteps to simulate.
    :type num_timesteps: int
    :param dirichlet_values: Fixed concentration values at boundary points (for Dirichlet conditions).
                             If `None`, Neumann (no-flux) conditions are applied.
    :type dirichlet_values: numpy.ndarray, optional
    :param boundary_type: Type of boundary conditions.
    :type boundary_type: str
    :param diffusion_const: Diffusion constant `D`. Default is 2.34e5.
    :type diffusion_const: float, optional
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

    **Example usage:**
    .. code-block:: python

        nx, ny, nz = 10, 10, 10

        init_condition = np.random.rand(nx, ny, nz)

        boundary = np.zeros((nx, ny, nz), dtype=bool)

        boundary[:, 0, :] = boundary[:, -1, :] = True

        result = diffusion_3d(
            init_condition, boundary, grid_spacing=1.0, timestep=0.01, num_timesteps=100
        )
    """

    nx, ny, nz = init_condition.shape

    ϕ: np.ndarray = np.zeros((nx, ny, nz, num_timesteps), dtype=np.float64)
    d = diffusion_const * timestep / grid_spacing**2

    ϕ[:, :, :, 0] = np.copy(init_condition)

    for it in range(1, num_timesteps):
        for ix in range(0, nx):
            for iy in range(0, ny):     
                for iz in range(0, nz):
                    if not boundary[ix, iy, iz]:
                        ϕ[ix, iy, iz, it] = (ϕ[ix, iy, iz, it - 1] + 0.25 * d * (
                                ϕ[(ix + 1) % nx, iy, iz, it - 1] +
                                ϕ[(ix - 1) % nx, iy, iz, it - 1] +
                                ϕ[ix, (iy + 1) % ny, iz, it - 1] +
                                ϕ[ix, (iy - 1) % ny, iz, it - 1] +
                                ϕ[ix, iy, (iz + 1) % nz, it - 1] +
                                ϕ[ix, iy, (iz - 1) % nz, it - 1] - 6 * ϕ[ix, iy, iz, it - 1]) +
                                reaction_rate * ϕ[ix, iy, iz, it - 1] * timestep)
                    else:
                        # Neumann Boundary Conditions
                        if boundary_type == 'neumann':
                            ϕ[ix, iy, iz, it] = ϕ[ix, iy, iz, it - 1]

                        # Dirichlet Boundary Conditions
                        elif boundary_type == 'dirichlet' and dirichlet_values is not None:
                            ϕ[ix, iy, iz, it] = dirichlet_values[ix, iy, iz]

                        else:
                            # Robin Boundary Conditions Here
                            ϕ[ix, iy, iz, it] = ϕ[ix, iy, iz, it - 1]
    return ϕ


def smash_fuel_rods_together(fuel_rod_1, fuel_rod_2, timestep: float, compression_speed: float) -> None:
    distance = fuel_rod_2.x_center - fuel_rod_1.x_center

    # Make sure the fuel rods are not inside eachother // overlapping
    if distance > 0:
        move_distance = min(compression_speed * timestep, distance)
        fuel_rod_1.x_center += move_distance
        fuel_rod_2.x_center -= move_distance
    else:
        print('Rods have collided')

    # return fuel_rod_1, fuel_rod_2


def update_neutron_density(init_condition_1, init_condition_2, fuel_rod_1, fuel_rod_2, time_step):
    # As the rods get closer, increase neutron density in the interacting region
    interaction_radius = 5  # Define an interaction radius that grows as rods compress

    # Find the region where the rods overlap or are close to each other
    distance_between_rods = abs(fuel_rod_2.x_center - fuel_rod_1.x_center)

    if distance_between_rods < interaction_radius:
        # Increase neutron density in the interaction zone
        init_condition_1[:, :,
        int(fuel_rod_1.x_center - interaction_radius):int(fuel_rod_1.x_center + interaction_radius)] += 0.1
        init_condition_2[:, :,
        int(fuel_rod_2.x_center - interaction_radius):int(fuel_rod_2.x_center + interaction_radius)] += 0.1

    return init_condition_1, init_condition_2
