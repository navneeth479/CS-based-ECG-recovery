import math

import Grid as grid
import flat_panel_project_utils as utils

phantom = utils.shepp_logan(512)

grid_test = grid.Grid(512, 512, 1)
origin = grid_test.get_origin()
# grid_test.set_buffer(phantom)
# val = grid_test.get_at_index(125, 340)

# ******* testing grid class to draw a circle **********
radius = 150
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


print("completed circle", grid_test.get_buffer())
utils.show(grid_test.get_buffer(), "test")

print("get at physical", grid.get_at_physical(grid_test, 200, 150))
print("value @", grid_test.get_buffer()[200][150])
# ******* circle test ended **********

# print(phantom[125][340])
# print(val)

