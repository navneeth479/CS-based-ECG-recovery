import numpy as np


class Grid:
    o_pixel = np.array([0, 0])

    def __init__(self, height, width, spacing):
        self.height = height
        self.width = width
        self.spacing = spacing
        self.buffer = np.empty([height, width], dtype=float)

    def get_origin(self):
        o_wcs = Grid.o_pixel
        o_wcs[0] = -0.5*(self.width - 1) * self.spacing
        o_wcs[1] = -0.5*(self.height - 1) * self.spacing
        return o_wcs

    def get_size(self):
        size = np.array([self.width, self.height])
        return size

    def get_spacing(self):
        spacing = np.array([self.spacing, self.spacing])
        return spacing

    def set_buffer(self, nd_array):
        self.buffer = nd_array

    def get_buffer(self):
        return self.buffer

    def index_to_physical(self, i, j):
        return np.array([i * self.spacing + Grid.o_pixel[0], j * self.spacing + Grid.o_pixel[1]])

    def physical_to_index(self, x, y):
        return np.array([(x - Grid.o_pixel[0])/self.spacing, (y - Grid.o_pixel[1])/self.spacing])

    def set_at_index(self):








