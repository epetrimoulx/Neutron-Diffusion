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


def plot_diffusion(time_step, y_index):
    """Plot a 2D slice of the 3D diffusion result at a specific time step."""
    plt.figure(figsize=(8, 6))
    
    # Create the imshow plot and store the return object
    im = plt.imshow(
        np.transpose(result[:, y_index, :, time_step]),
        cmap='coolwarm',
        origin='lower',
        vmin=0.0,
        vmax=3e33
    )
    
    # Add contour on top of the image
    plt.contour(
        np.transpose(result[:, y_index, :, time_step]),
        levels=30,
        linewidths=0.5,
        colors='k',
        alpha=0.5
    )
    
    # Use the imshow object to add the color bar
    plt.colorbar(im, fraction=0.02)
    
    plt.title(f"Diffusion at step {time_step}")
    plt.xlabel("X-axis")
    plt.ylabel("Z-axis")
    plt.show()


def plot_k_vs_time(k, time):
    plt.plot(time[:], k[:-1])
    plt.xlabel('Time (s)')
    plt.ylabel('Neutron Multiplication Factor (k)')
    plt.title('Neutron Multiplication Factor vs Time')
    plt.grid(True)
    plt.show()