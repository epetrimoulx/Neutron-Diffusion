class Shape:
    def __init__(self, length, width, height, name, mass, element_name, element_num):
        self.name = name
        self.length = length
        self.width = width
        self.height = height
        self.mass = mass
        self.element = element_name
        self.element_num = element_num

    def __str__(self):
        return (f'Element: {self.element}-{self.mass}\n'
                f'Z: {self.element_num}\n'
                f'Shape: {self.name}\n'
                f'Length: {self.length}\n'
                f'Width: {self.width}\n'
                f'Height: {self.height}')

    def calc_num_neutrons(self):
        return self.mass - self.element_num


