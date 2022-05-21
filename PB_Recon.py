import Grid as grid
import numpy as np
import Helpers.Utility_functions as helper


def create_sinogram(phantom, projections, detector_spacing, detector_size, angular_scan_range):

    grid_phantom = grid.Grid(len(phantom), len(phantom[0]), (2, 2))
    grid_phantom.set_buffer(phantom)
    angular_step = angular_scan_range / projections
    detector_length = detector_spacing * detector_size

    maxThetaIndex = int(angular_scan_range / angular_step)
    maxSIndex = int(detector_length / detector_spacing)

    N = grid_phantom.get_size()
    points = int(np.ceil(2 * np.sqrt(2) * N[0]))
    theta = np.linspace(0, angular_scan_range, projections)
    s = np.linspace(-np.sqrt(2), np.sqrt(2), points)
    grid_sino = grid.Grid(maxSIndex, maxThetaIndex, (angular_step, detector_spacing))
    grid_sino.set_origin(0, maxSIndex / 2)
    for i in range(points):
        for j in range(projections - 1):
            grid_sino.set_at_index(i, j, helper.line_integral(grid_phantom, s[i], theta[j]))
    return grid_sino
