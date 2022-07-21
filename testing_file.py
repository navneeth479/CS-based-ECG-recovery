import flat_panel_project_utils as utils
import PB_Recon as recon
import numpy as np
import Grid as grid
import time

#### Test for unfiltered backprojection ######

phantom = utils.shepp_logan(128)
grid_phantom = grid.Grid(len(phantom), len(phantom[0]), (1, 1))
grid_phantom.set_buffer(phantom)
utils.show(grid_phantom.get_buffer(), " Phantom ")

sinogram = recon.create_sinogram(grid_phantom.get_buffer(), 300, 1, 200, np.pi)
utils.show(np.rot90(sinogram.get_buffer(), k=1), "Sinogram ")

filtered_sino = recon.ramp_filter(sinogram, 1)
utils.show(np.rot90(filtered_sino.get_buffer(), k=1), "Sinogram filtered")

start_time = time.time()
recon_image, library = recon.backproject(filtered_sino, grid_phantom.get_size()[0], grid_phantom.get_size()[1],grid_phantom.get_spacing())
utils.show(recon_image, "FBP image ")
print(" CPU backprojection run time (s) : %s" % (time.time() - start_time))

#utils.show(grid_phantom.get_buffer() - recon_image_unfiltered, "Error image (phantom v backprojection) ")
#utils.show(grid_phantom.get_buffer() - np.rot90(library, k=-1), "Error image (Phantom v library)")

start_time = time.time()
img_opencl = recon.backprojectOpenCL(filtered_sino, grid_phantom.get_size()[0], grid_phantom.get_size()[1], grid_phantom.get_spacing())
utils.show(np.rot90(library, k=-1), "OpenCL recon image")
print("  GPU accelerated backprojection run time(s) : %s" % (time.time() - start_time))
