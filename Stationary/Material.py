class Shape:
    def __init__(self, length: int, width: int, height: int, name: str, mass: float, element_name: str, element_num: int, density: float):
        self.name: str = name
        self.length: int = length
        self.width: int = width
        self.height: int = height
        self.element: str = element_name
        self.element_num: int = element_num
        self.density: float = density
        self.mass: float = mass * AU_to_Kg

    def __str__(self):
        return (f'Element: {self.element}-{int(self.mass / AU_to_Kg)}\n'
                f'Z: {self.element_num}\n'
                f'ρ: {self.density} kg/m^3\n'
                f'M: {self.mass:.2e} kg\n'
                f'Shape: {self.name}\n'
                f'Length: {self.length}m\n'
                f'Width: {self.width}m\n'
                f'Height: {self.height}m')

    def calc_num_neutrons(self):
        return (self.mass / AU_to_Kg) - self.element_num

    def set_initial_conditions(self):
        initial_condition = np.zeros((self.length, self.width, self.height))

        if self.name == "Cylinder":
            cylinder_radius = 3  # Radius of the cylinder
            volume = np.pi * cylinder_radius / self.height
            cylinder_center = (self.length // 2, self.width // 2)  # Center of the cylinder in the XY plane
            cylinder_axis = 'z'  # Axis of the cylinder ('x', 'y', or 'z')

            # Fill the cylinder within the cube
            for x in range(self.length):
                for y in range(self.width):
                    for z in range(self.height):
                        if cylinder_axis == 'z':  # Cylinder aligned along the z-axis
                            distance = np.sqrt((x - cylinder_center[0]) ** 2 + (y - cylinder_center[1]) ** 2)
                            if distance <= cylinder_radius and z < self.height:
                                initial_condition[x, y, z] = self.density * volume * self.calc_num_neutrons() / self.mass

        elif self.name == "Sphere":
            radius = 3
            volume = 4/3 * np.pi * radius**3
            x0, y0, z0 = int(np.floor(initial_condition.shape[0] / 2)), int(np.floor(initial_condition.shape[1] / 2)), int(np.floor(initial_condition.shape[2] / 2))
            for x in range(x0 - radius, x0 + radius + 1):
                for y in range(y0 - radius, y0 + radius + 1):
                    for z in range(z0 - radius, z0 + radius + 1):
                        ''' deb: measures how far a coordinate in A is far from the center. 
                                deb>=0: inside the sphere.
                                deb<0: outside the sphere.'''
                        deb = radius - abs(x0 - x) - abs(y0 - y) - abs(z0 - z)
                        if deb >= 0:
                            initial_condition[x, y, z] = self.density * volume * self.calc_num_neutrons() / self.mass

        elif self.name == "Triangular Prism":
            volume = self.length * self.width / 2 * self.height
            for x in range(0, self.length):
                for y in range(x, self.width):
                    for z in range(y, self.height):
                        initial_condition[x, y, z] = self.density * volume * self.calc_num_neutrons() / self.mass

        elif self.name == "Left Hemisphere":
            radius = 3
            volume = 2 / 3 * np.pi * radius**3 
            x0, y0, z0 = int(np.floor(initial_condition.shape[0] / 2)), int(
                np.floor(initial_condition.shape[1] / 2)), int(np.floor(initial_condition.shape[2] / 2))
            for x in range(x0 - radius, x0 + radius + 1):
                for y in range(y0 - radius, y0 + radius + 1):
                    for z in range(z0 - radius, z0 + radius + 1):
                        deb = radius - abs(x0 - x) - abs(y0 - y) - abs(z0 - z)
                        if deb >= 0:
                            initial_condition[x, y, z] = self.density * volume * self.calc_num_neutrons() / self.mass
            initial_condition[:, :, self.height // 2:] = 0.0

        elif self.name == "Right Hemisphere":
            radius = 3
            volume = 2 / 3 * np.pi * radius**3 
            x0, y0, z0 = int(np.floor(initial_condition.shape[0] / 2)), int(
                np.floor(initial_condition.shape[1] / 2)), int(np.floor(initial_condition.shape[2] / 2))
            for x in range(x0 - radius, x0 + radius + 1):
                for y in range(y0 - radius, y0 + radius + 1):
                    for z in range(z0 - radius, z0 + radius + 1):
                        deb = radius - abs(x0 - x) - abs(y0 - y) - abs(z0 - z)
                        if deb >= 0:
                            initial_condition[x, y, z] = self.density * volume * self.calc_num_neutrons() / self.mass

            initial_condition[:, :, :self.height // 2] = 0.0

        else:
            volume = self.length * self.width * self.height
            initial_condition.fill(self.density * volume * self.calc_num_neutrons() / self.mass)

        return initial_condition