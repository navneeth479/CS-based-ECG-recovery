import numpy as np
import flat_panel_project_utils as utils


class Grid:

    def __init__(self, height, width, spacing):
        self.height = height
        self.width = width
        self.spacing = spacing
        self.origin = [-0.5 * (width - 1.0) * spacing[0], -0.5 * (height - 1.0) * spacing[1]]
        self.buffer = np.empty([height, width], dtype=float)

    def get_origin(self):
        return self.origin

    def set_origin(self, o_x, o_y):
        self.origin = o_x, o_y

    def get_size(self):
        size = np.array([self.width, self.height])
        return size

    def get_spacing(self):
        spacing = np.array([self.spacing[0], self.spacing[1]])
        return spacing

    def set_buffer(self, nd_array):
        self.buffer = nd_array.copy()

    def get_buffer(self):
        return self.buffer

    def index_to_physical(self, i, j):
        return np.array([(i * self.spacing[0]) + self.origin[0], (j * self.spacing[1]) + self.origin[1]])

    def physical_to_index(self, x, y):
        return np.array([((x - self.origin[0]) / self.spacing[0]),
                         ((y - self.origin[1]) / self.spacing[1])])

    def set_at_index(self, i, j, val):
        self.buffer[i][j] = val

    def add_at_index(self, i, j, val):
        self.buffer[i][j] += val

    def get_at_index(self, i, j):
        return self.buffer[i][j]

    def get_at_physical(self, x, y):
        index = Grid.physical_to_index(self, x, y)
        i, j = index[0], index[1]
        return utils.interpolate(self, i, j)
