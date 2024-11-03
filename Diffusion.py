"""
File to deal with the Diffusion Equation
"""

import numba
import numpy as np

@numba.njit
def diffusion_3d(init_condition, boundary, diffusion_const, grid_spacing, timestep, num_timesteps):
    nx, ny, nz = init_condition.shape

    ϕ = np.zeros((nx, ny, nz, num_timesteps))
    d = diffusion_const * timestep / grid_spacing**2

    ϕ[:, :, :, 0] = np.copy(init_condition)

    for it in range(1, num_timesteps):
        for ix in range(0, nx):
            for iy in range(0, ny):     
                for iz in range(0, nz):
                    if not boundary[ix, iy, iz]:
                        ϕ[ix, iy, iz, it] = ϕ[ix, iy, iz, it - 1] + 0.25 * d * (
                                ϕ[(ix + 1) % nx, iy, iz, it - 1] +
                                ϕ[(ix - 1) % nx, iy, iz, it - 1] +
                                ϕ[ix, (iy + 1) % ny, iz, it - 1] +
                                ϕ[ix, (iy - 1) % ny, iz, it - 1] +
                                ϕ[ix, iy, (iz + 1) % nz, it - 1] +
                                ϕ[ix, iy, (iz - 1) % nz, it - 1] - 6 * ϕ[ix, iy, iz, it - 1])
                    else:
                        ϕ[ix, iy, iz, it] = ϕ[ix, iy, iz, it - 1]
    return ϕ