import math
import Grid as grid
import flat_panel_project_utils as utils

# **** TEST - 1 ****
# ******* testing grid class on Shepp Logan phantom **********
phantom = utils.shepp_logan(512)

# note: constructor params are -> height, width, (s_x, s_y), (o_x, o_y)
# spacing and origin are to be set as tuples
grid_phantom = grid.Grid(len(phantom), len(phantom[0]), (1, 1), (0, 0))
grid_phantom.set_buffer(phantom)
utils.show(grid_phantom.get_buffer(), "Shepp Logan Phantom ")

print("Origin is set at:", grid_phantom.get_origin())
print("Spacing values are:", grid_phantom.get_spacing())
print("Grid size is:", grid_phantom.get_size())
print("get at physical of phantom @",(150, 157),"is :", grid_phantom.get_at_physical(grid_phantom, 150, 157))

# *******  END **********


# **** TEST - 2 ****
# ******* Using grid class methods to draw a circle **********
grid_test = grid.Grid(512, 512, (1, 1), (-255, -255))
origin = grid_test.get_origin()

radius = 100
val = 100
width = grid_test.get_size()[0]
height = grid_test.get_size()[1]
for x in range(-radius, radius):
    for y in range(-radius, radius):
        if (math.pow(x, 2) + math.pow(y, 2)) <= math.pow(radius, 2):
            index = grid_test.physical_to_index(x, y)
            i, j = index[0].astype('int64'), index[1].astype('int64')

            if 0 <= i < width and 0 <= j < height:
                grid_test.set_at_index(i, j, val)

utils.show(grid_test.get_buffer(), "test")

print("get at physical @",(20, 20),"is :", grid_test.get_at_physical(grid_test, 0, 100))
# ******* circle test ended **********
