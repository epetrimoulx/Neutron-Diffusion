class Shape:
    """
    Note: the length, width, and height members are described in terms of # of Uranium atoms

    Ex. length = 5 means 5 Uranium atoms in length
    """
    def __init__(self, length: int, width: int, height: int, name: str, mass: int, element_name: str, element_num: int):
        self.name: str = name
        self.length: int = length
        self.width: int = width
        self.height: int = height
        self.mass: int = mass
        self.element: str = element_name
        self.element_num: int = element_num

    def __str__(self):
        return (f'Element: {self.element}-{self.mass}\n'
                f'Z: {self.element_num}\n'
                f'Shape: {self.name}\n'
                f'Length: {self.length}\n'
                f'Width: {self.width}\n'
                f'Height: {self.height}')

    def calc_num_neutrons(self):
        return self.mass - self.element_num
