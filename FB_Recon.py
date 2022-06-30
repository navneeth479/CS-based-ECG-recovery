import Grid as grid
import math
from skimage.transform import iradon
import numpy as np
import flat_panel_project_utils as utils


def create_fanogram(phantom, projections, detector_spacing, detector_sizeInPixels, angular_increment, d_si, d_sd):
    # Improvements : Intersection of rays with object can help reduce loop size for sampling along rays
    grid_phantom = grid.Grid(len(phantom), len(phantom[0]), (1, 1))
    grid_phantom.set_buffer(phantom)
    angular_scan_range = angular_increment * projections
    detector_length = detector_spacing * detector_sizeInPixels

    gamma = np.arctan(detector_length / d_sd)
    print("fan angle is: ", np.rad2deg(gamma))
    beta = np.linspace(0, np.deg2rad(angular_scan_range) + gamma, projections)
    new_angular_increment = (angular_scan_range + np.rad2deg(gamma)) / projections
    d_id = d_sd - d_si
    num_of_Angles = len(beta)
    t = np.linspace(-(detector_length - 1) * detector_spacing / 2, (detector_length - 1) * detector_spacing / 2,
                    detector_sizeInPixels)
    grid_fano = grid.Grid(detector_sizeInPixels, num_of_Angles, (new_angular_increment, detector_spacing))
    grid_fano.set_origin(0, -(detector_length - 1) * detector_spacing / 2)
    p1_source = [d_si, 0]

    for i in range(num_of_Angles):
        # calculate Source position
        S = np.array([-p1_source[0] * np.sin(beta[i]) + p1_source[1] * np.cos(beta[i]),
                      p1_source[0] * np.cos(beta[i]) + p1_source[1] * np.sin(beta[i])])
        for j in range(detector_sizeInPixels):
            # calculate detector position along detector
            P = np.array([d_id * np.sin(beta[i]) + t[j] * np.cos(beta[i]),
                          - d_id * np.cos(beta[i]) + t[j] * np.sin(beta[i])])

            SP_len = math.floor(np.sqrt(np.power(P[0] - S[0], 2) + np.power(P[1] - S[1], 2)))
            x, y = np.linspace(S[0], P[0], SP_len), np.linspace(S[1], P[1], SP_len)
            ray_sum = 0

            for pos in range(SP_len):
                ray_sum += grid_phantom.get_at_physical(x[pos], y[pos])
                # ray_sum += utils.interpolate(grid_phantom, x[pos], y[pos])
            grid_fano.set_at_index(j, i, ray_sum)

    return grid_fano


def rebinning(fanogram, d_si, d_sd):
    detector_length = fanogram.get_size()[1]
    deltaS = fanogram.get_spacing()[1]
    projections = len(fanogram.get_buffer()[0])
    deltaTheta = 180 / projections
    theta = np.linspace(0, np.deg2rad(180), projections, endpoint=False)
    grid_sino = grid.Grid(detector_length, projections, (deltaTheta, deltaS))
    grid_sino.set_origin(0, -(detector_length - 1) * deltaS / 2)
    s = np.linspace(0, (detector_length - 1) * deltaS,
                    detector_length)

    for i in range(detector_length):
        for j in range(0, projections):
            gamma = np.arctan(detector_length / d_sd)
            beta = theta - gamma
            t = s / np.cos(gamma)

            beta_sorted = np.where(beta < 0, beta + np.pi, beta)
            beta_sorted = np.sort(beta_sorted)

            proj_index = np.rad2deg(beta_sorted[j]) / deltaTheta

            if fanogram.get_size()[0] > proj_index >= 0 and fanogram.get_size()[1] > t[i] >= 0:
                #val = fanogram.get_at_index(int(t[i]), int(proj_index))
                val = utils.interpolate(fanogram, t[i], proj_index)
                grid_sino.set_at_index(i, j, val)

    return grid_sino


def backproject(sinogram, recon_size_x, recon_size_y, spacing):
    # Not being used for reconstructing rebinned projections.
    # To do:- Implement FBP for fab beam equidistant case.

    recon_img = grid.Grid(recon_size_x, recon_size_y, spacing)  # spacing should be received as a tuple
    detector_length = sinogram.get_size()[1]
    deltaS = sinogram.get_spacing()[1]
    projections = len(sinogram.get_buffer()[0])

    theta = np.linspace(0, 180, projections, endpoint=False)
    img_fbp = iradon(sinogram.get_buffer(), theta=theta, filter_name=None, circle=False)
    return img_fbp
