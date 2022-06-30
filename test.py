import numpy as np
import Grid as grid
import flat_panel_project_utils as utils
import PB_Recon as recon
import FB_Recon as recon1



# **** TEST - 1 ****
# ******* testing grid class on Shepp Logan phantom **********
phantom = utils.shepp_logan(64)
# note: constructor params are -> height, width, (s_x, s_y), (o_x, o_y)
# spacing and origin are to be set as tuples
grid_phantom = grid.Grid(len(phantom), len(phantom[0]), (1, 1))
grid_phantom.set_buffer(phantom)
utils.show(grid_phantom.get_buffer(), " Phantom ")

#sinogram = recon.create_sinogram(grid_phantom.get_buffer(), 200, 1, 250, np.pi)
fanogram = recon1.create_fanogram(grid_phantom.get_buffer(), 200, 1, 200, 0.9, 500, 1000)
utils.show(np.rot90(fanogram.get_buffer(),  k=1), "Fanogram ")


sinogram_rebinned = recon1.rebinning(fanogram, 500, 1000)
utils.show(np.rot90(sinogram_rebinned.get_buffer(),  k=1), "Sinogram ")

library = recon1.backproject(sinogram_rebinned, grid_phantom.get_size()[0], grid_phantom.get_size()[1], grid_phantom.get_spacing())
utils.show(np.rot90(library, k=-1), "unfiltered backprojection")

sino_filtered = recon.ramp_filter(sinogram_rebinned, fanogram.get_spacing()[1])
utils.show(np.rot90(sino_filtered.get_buffer(), k=1), "Filtered Sinogram ")

library = recon1.backproject(sino_filtered, grid_phantom.get_size()[0], grid_phantom.get_size()[1], grid_phantom.get_spacing())
utils.show(np.rot90(library, k=-1), "Filtered backprojection")
# ---------------------------------------------------------------------------
#recon_image_unfiltered, library = recon.backproject(sinogram, grid_phantom.get_size()[0], grid_phantom.get_size()[1], grid_phantom.get_spacing())
#utils.show(recon_image_unfiltered, "unfiltered ")
#utils.show(grid_phantom.get_buffer() - recon_image_unfiltered, "Error image 1")
# -----------------------------------------------------------------------------

#sino_filtered = recon.ramp_filter(sinogram, sinogram.get_spacing()[1])
#utils.show(np.rot90(sino_filtered.get_buffer(), k=1), "Filtered Sinogram ")


#utils.show(grid_phantom.get_buffer() - recon_image, "Error image (Phantom v recon)")
#utils.show(grid_phantom.get_buffer() - np.rot90(library, k=-1), "Error image (Phantom v library)")


# *******  END **********


