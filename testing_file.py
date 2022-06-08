import flat_panel_project_utils as utils
import PB_Recon as recon
import numpy as np
import Grid as grid

#### Test for unfiltered backprojection ######

phantom = utils.shepp_logan(64)
grid_phantom = grid.Grid(len(phantom), len(phantom[0]), (1, 1))
grid_phantom.set_buffer(phantom)
utils.show(grid_phantom.get_buffer(), " Phantom ")

sinogram = recon.create_sinogram(grid_phantom.get_buffer(), 200, 1, 250, np.pi)
utils.show(np.rot90(sinogram.get_buffer(),  k=1), "Sinogram ")

recon_image_unfiltered, library = recon.backproject(sinogram, grid_phantom.get_size()[0], grid_phantom.get_size()[1], grid_phantom.get_spacing())

utils.show(recon_image_unfiltered, "unfiltered reconstruction image ")
utils.show(grid_phantom.get_buffer() - recon_image_unfiltered, "Error image (phantom v backprojection) ")
utils.show(np.rot90(library, k=-1), "FBP image using iradon library ")
utils.show(grid_phantom.get_buffer() - np.rot90(library, k=-1), "Error image (Phantom v library)")