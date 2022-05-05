import math

import Grid as grid
import flat_panel_project_utils as utils
import matplotlib.pyplot as plt

phantom = utils.shepp_logan(512)
grid_phantom = grid.Grid(256,256,1)
grid_phantom.set_buffer(phantom)
utils.show(grid_phantom.get_buffer(), "Phantom ")

# val = grid_test.get_at_index(125, 340)

# ******* testing grid class to draw a circle **********
grid_test = grid.Grid(512, 512, 4)
origin = grid_test.get_origin()

radius = 100
val = 100
width = grid_test.get_size()[0]
height = grid_test.get_size()[1]
for x in range(-radius, radius):
    for y in range(-radius, radius):
        if (math.pow(x, 2) + math.pow(y, 2)) <= math.pow(radius, 2):
            index = grid_test.physical_to_index(x, y)
            i, j = index[0], index[1]

            if 0 <= i < width and 0 <= j < height:
                grid_test.set_at_index(i, j, val)
                print("values of Y", y)


print("completed circle", grid_test.get_buffer())
utils.show(grid_test.get_buffer(), "test")

print("get at physical", grid.get_at_physical(grid_test, 200, 150))
# ******* circle test ended **********


