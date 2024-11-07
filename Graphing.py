from matplotlib import pyplot as plt

def plot_neutron_density(neutron_density, time) -> None:
    plt.figure()
    plt.plot(time, neutron_density)
    plt.title('Neutron Density over time')
    plt.ylabel('Density of Neutrons [neutrons / unit volume]')
    plt.xlabel('Time')
    plt.show()

def plot_k_vs_time(k, time) -> None:
    plt.figure()
    plt.plot(time, k)
    plt.title('Neutron Multiplication Factor vs Time')
    plt.ylabel('Neutron Multiplication Rate')
    plt.xlabel('Time')
    plt.show()