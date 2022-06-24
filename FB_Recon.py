import Grid as grid
import math
from skimage.transform import iradon
import flat_panel_project_utils as utils
import numpy as np


def create_fanogram(phantom, projections, detector_spacing, detector_sizeInPixels, angular_increment, d_si, d_sd):
    grid_phantom = grid.Grid(len(phantom), len(phantom[0]), (1, 1))
    grid_phantom.set_buffer(phantom)
    angular_scan_range = angular_increment * projections
    detector_length = detector_spacing * detector_sizeInPixels

    gamma = np.arctan(detector_length / d_sd) * (np.pi / 180)
    beta = np.linspace(0, np.deg2rad(180) + 2 * gamma, projections)
    d_id = d_sd - d_si
    num_of_Angles = len(beta)
    t = np.linspace(-(detector_length - 1) * detector_spacing / 2, (detector_length - 1) * detector_spacing / 2,
                    detector_sizeInPixels)
    grid_sino = grid.Grid(detector_sizeInPixels, num_of_Angles, (angular_increment, detector_spacing))
    grid_sino.set_origin(0, -(detector_length - 1) * detector_spacing / 2)

    for i in range(num_of_Angles):
        for j in range(detector_sizeInPixels):
            S = np.array([-d_si * np.sin(beta[i]), d_si * np.cos(beta[i])])
            M = np.array([d_id * np.sin(beta[i]), -d_id * np.cos(beta[i])])
            MP = np.array([t[j] * np.cos(beta[i]), t[j] * np.sin(beta[i])])
            P = M + MP

            SP_len = math.sqrt((P[0] - S[0]) ** 2 + (P[1] - S[1]) ** 2)
            x, y = np.linspace(S[0], P[0], int(SP_len)), np.linspace(S[1], P[1], int(SP_len))
            ray_sum = 0

            for pos in range(len(x)):
                ray_sum += grid_phantom.get_at_physical(x[pos], y[pos])
                #ray_sum += utils.interpolate(grid_phantom, x[pos], y[pos])
            grid_sino.set_at_index(j, i, ray_sum)

    return grid_sino


def backproject(sinogram, recon_size_x, recon_size_y, spacing):
    recon_img = grid.Grid(recon_size_x, recon_size_y, spacing)  # spacing should be received as a tuple
    detector_length = sinogram.get_size()[1]
    deltaS = sinogram.get_spacing()[1]
    projections = len(sinogram.get_buffer()[0])

    theta = np.linspace(0, 180, projections, endpoint=False)
    img_fbp = iradon(sinogram.get_buffer(), theta=theta, filter_name=None, circle=False)
    return img_fbp
