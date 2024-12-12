from matplotlib import pyplot as plt
import numpy as np


def plot_fuel_rod_positions(array):
    # Get coordinates where the array is True
    x, y, z = np.where(array != 0)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Clear previous plot data
    ax.cla()

    # Plot the cubes as red points
    ax.scatter(x, y, z, c='r', marker='o')

    # Set limits for axes based on the grid size
    ax.set_xlim([0, array.shape[0] - 1])
    ax.set_ylim([0, array.shape[1] - 1])
    ax.set_zlim([0, array.shape[2] - 1])

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Fuel Rod Positions')

    # Optionally add a grid for better spatial orientation
    ax.grid(True)
    plt.show()


def plot_neutron_density_evolution(result, time, timestep_interval=10):
    for timestep in range(0, len(time), timestep_interval):
        plt.imshow(np.transpose(result[:, -1//2, :, timestep]), cmap='coolwarm', origin='lower', vmin=0)
        plt.colorbar(fraction=0.02)
        plt.title(f'Neutron Density at Time {time[timestep]}s')
        plt.show()


def plot_k_vs_time(k, time):
    plt.plot(time[:], k[:-1])
    plt.xlabel('Time (s)')
    plt.ylabel('Neutron Multiplication Factor (k)')
    plt.title('Neutron Multiplication Factor vs Time')
    plt.grid(True)
    plt.show()