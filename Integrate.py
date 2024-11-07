
def rk4(function, timestep, u, t):
    k1 = timestep * function(u, t)
    k2 = timestep * function(u + 0.5 * k1, t + 0.5 * timestep)
    k3 = timestep * function(u + 0.5 * k2, t + 0.5 * timestep)
    k4 = timestep * function(u + k3, t + timestep)

    return u + (k1 + 2*(k2 + k3) + k4) / 6.0