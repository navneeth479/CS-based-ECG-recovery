import math
import numpy as np
import Grid as grid
import flat_panel_project_utils as utils
import PB_Recon as recon
import matplotlib.pyplot as plt

# **** TEST - 1 ****
# ******* testing grid class on Shepp Logan phantom **********
phantom = utils.shepp_logan(64)

# note: constructor params are -> height, width, (s_x, s_y), (o_x, o_y)
# spacing and origin are to be set as tuples
grid_phantom = grid.Grid(len(phantom), len(phantom[0]), (2, 2))
grid_phantom.set_buffer(phantom)
utils.show(grid_phantom.get_buffer(), "Shepp Logan Phantom ")

sinogram = recon.create_sinogram(grid_phantom.get_buffer(), 500, 1, 250, np.pi)
#utils.show(sinogram.get_buffer(), "Sinogram ")
plt.imshow(sinogram.get_buffer())
plt.title("Sinogram")
plt.gray()
plt.show()
print("Sinogram is", sinogram.get_buffer())

print(" indices of:", (21, 40), "are:", grid_phantom.physical_to_index(21, 30) )
print("get at physical of phantom", grid_phantom.get_at_physical(21, 40))

# *******  END **********


# **** TEST - 2 ****
# ******* Using grid class methods to draw a circle **********
grid_test = grid.Grid(512, 512, (1, 1))
origin = grid_test.get_origin()

radius = 50
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


utils.show(grid_test.get_buffer(), "circular phantom")


#print("get at physical @",(20, 20),"is :", grid_test.get_at_physical(20, 20))
# ******* circle test ended **********
