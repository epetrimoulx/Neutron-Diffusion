@numba.njit
def diffusion_3d(
        init_condition: np.ndarray,
        grid_spacing: float,
        timestep: float,
        num_timesteps: int,
        boundary: np.ndarray,
        dirichlet_values: np.ndarray = None,
        boundary_type: str = 'dirichlet',
        diffusion_const: float = 2.34e5,
        reaction_rate: float = 1.896e8
) -> np.ndarray:
    """
    Simulates the 3D diffusion-reaction process over a grid with Dirichlet or Neumann boundary conditions.
    This function evolves the neutron density `n` over time according to the diffusion-reaction equation:
        ∂n/∂t = D ∇²n + ηn
    where `D` is the diffusion constant, and η is the reaction rate constant.
    
    :param init_condition: Initial neutron density array (3D ndarray).
    :type init_condition: numpy.ndarray
    :param grid_spacing: Distance between grid points in the spatial domain.
    :type grid_spacing: float
    :param timestep: Time increment for each simulation step.
    :type timestep: float
    :param num_timesteps: Number of timesteps to simulate.
    :type num_timesteps: int
    :param boundary: Boolean 3D array marking boundary (`True`) and interior (`False`) points.
    :type boundary: numpy.ndarray
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

    diffusion: np.ndarray = np.zeros((nx, ny, nz, num_timesteps), dtype=np.float64)
    d = diffusion_const * timestep / grid_spacing**2

    diffusion[:, :, :, 0] = np.copy(init_condition)

    for it in range(1, num_timesteps):
        for ix in range(0, nx):
            for iy in range(0, ny):     
                for iz in range(0, nz):
                    if not boundary[ix, iy, iz]:
                        if (ix < nx // 2 + 3) and (ix > nx // 2 - 3):
                            diffusion[ix, iy, iz, it] = (diffusion[ix, iy, iz, it - 1] + 0.25 * d * (
                                    diffusion[(ix + 1) % nx, iy, iz, it - 1] +
                                    diffusion[(ix - 1) % nx, iy, iz, it - 1] +
                                    diffusion[ix, (iy + 1) % ny, iz, it - 1] +
                                    diffusion[ix, (iy - 1) % ny, iz, it - 1] +
                                    diffusion[ix, iy, (iz + 1) % nz, it - 1] +
                                    diffusion[ix, iy, (iz - 1) % nz, it - 1] - 6 * diffusion[ix, iy, iz, it - 1]) +
                                    reaction_rate * diffusion[ix, iy, iz, it - 1] * timestep)
                        else:
                            diffusion[ix, iy, iz, it] = (diffusion[ix, iy, iz, it - 1] + 0.25 * d * (
                                    diffusion[(ix + 1) % nx, iy, iz, it - 1] +
                                    diffusion[(ix - 1) % nx, iy, iz, it - 1] +
                                    diffusion[ix, (iy + 1) % ny, iz, it - 1] +
                                    diffusion[ix, (iy - 1) % ny, iz, it - 1] +
                                    diffusion[ix, iy, (iz + 1) % nz, it - 1] +
                                    diffusion[ix, iy, (iz - 1) % nz, it - 1] - 6 * diffusion[ix, iy, iz, it - 1]))

                    else:
                        # Neumann Boundary Conditions (zero flux)
                        if boundary_type == 'neumann':
                            if ix == 0:
                                diffusion[ix, iy, iz, it] = diffusion[ix + 1, iy, iz, it - 1]
                            elif ix == nx - 1:
                                diffusion[ix, iy, iz, it] = diffusion[ix - 1, iy, iz, it - 1]
                            if iy == 0:
                                diffusion[ix, iy, iz, it] = diffusion[ix, iy + 1, iz, it - 1]
                            elif iy == ny - 1:
                                diffusion[ix, iy, iz, it] = diffusion[ix, iy - 1, iz, it - 1]
                            if iz == 0:
                                diffusion[ix, iy, iz, it] = diffusion[ix, iy, iz + 1, it - 1]
                            elif iz == nz - 1:
                                diffusion[ix, iy, iz, it] = diffusion[ix, iy, iz - 1, it - 1]

                        # Dirichlet Boundary Conditions
                        elif boundary_type == 'dirichlet' and dirichlet_values is not None:
                            diffusion[ix, iy, iz, it] = dirichlet_values[ix, iy, iz]

                        else:
                            # Robin Boundary Conditions Here
                            diffusion[ix, iy, iz, it] = diffusion[ix, iy, iz, it - 1]
    return diffusion