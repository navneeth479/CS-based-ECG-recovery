import numpy as np
import flat_panel_project_utils as utils


class Grid:
    o_pixel = np.array([0, 0])

    def __init__(self, height, width, spacing):
        self.height = height
        self.width = width
        self.spacing = spacing
        self.origin_p = np.array([0, 0])
        self.buffer = np.empty([height, width], dtype=float)

    def set_origin(self):
        self.o_wcs = self.origin_p
        self.o_wcs[0], self.o_wcs[1] = -0.5 * (self.width - 1) * self.spacing, -0.5 * (self.height - 1) * self.spacing
        return self.o_wcs

    def get_origin(self):
        return self.o_wcs

    def get_size(self):
        size = np.array([self.width, self.height])
        return size

    def get_spacing(self):
        spacing = np.array([self.spacing, self.spacing])
        return spacing

    def set_buffer(self, nd_array):
        self.buffer = nd_array.copy()

    def get_buffer(self):
        return self.buffer

    def index_to_physical(self, i, j):
        origin = Grid.get_origin(self)
        return np.array([(i * self.spacing) + origin[0], (j * self.spacing) + origin[1]])

    def physical_to_index(self, x, y):
        return np.array([((x - self.origin_p[0]) / self.spacing).astype('int64'),
                         ((y - self.origin_p[1]) / self.spacing).astype('int64')])

    def set_at_index(self, i, j, val):
        if i >= Grid.get_size(self)[0] or j >= Grid.get_size(self)[1]:
            print("indices are out of range")
        else:
            self.buffer[i][j] = val

    def get_at_index(self, i, j):
        return self.buffer[i][j]


    def get_at_physical(self, grid, x, y):
        return utils.interpolate(grid, x, y)