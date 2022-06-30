import Grid as grid
from skimage.transform import iradon
import flat_panel_project_utils as utils
import numpy as np
import matplotlib.pyplot as plt
import Helpers.Utility_functions as helper
from scipy.fftpack import fft, ifft, fftfreq


def create_sinogram(phantom, projections, detector_spacing, detector_size, angular_scan_range):
    grid_phantom = grid.Grid(len(phantom), len(phantom[0]), (1, 1))
    grid_phantom.set_buffer(phantom)
    angular_step = angular_scan_range / projections
    detector_length = detector_spacing * detector_size

    theta = np.linspace(0, angular_scan_range, projections)
    num_of_Angles = len(theta)
    s = np.linspace(-(detector_length - 1) * detector_spacing / 2, (detector_length - 1) * detector_spacing / 2,
                    detector_size)
    grid_sino = grid.Grid(detector_size, num_of_Angles, (angular_step, detector_spacing))
    grid_sino.set_origin(0, -(detector_length - 1) * detector_spacing / 2)
    for i in range(detector_size):
        for j in range(num_of_Angles):
            grid_sino.set_at_index(i, j, helper.line_integral(grid_phantom, s[i], theta[j]))
    return grid_sino


def ramp_filter(sinogram, detector_spacing):
    # Zero padding calculation and calculating ramp filter kernel
    projection_size_padded = helper.next_power_of_two(sinogram.buffer.shape[0])
    pad_width = np.zeros([(projection_size_padded - sinogram.buffer.shape[0]) // 2, sinogram.buffer.shape[1]])
    padded_sino = np.vstack((pad_width, sinogram.buffer))
    padded_sino = np.vstack((padded_sino, pad_width))
    delta_f = 1 / (detector_spacing * projection_size_padded)
    f = fftfreq(projection_size_padded, d=delta_f).reshape(-1, 1)
    fourier_filter = 2 * np.abs(f)

    # Optional : plotting the filter kernel in freq domain

    # Apply ramp filter in Fourier domain. Filtering row wise (along each detector line)
    sino_filtered = grid.Grid(padded_sino.shape[0], padded_sino.shape[1],
                              (sinogram.get_spacing()[0], sinogram.get_spacing()[1]))
    sino_fft = fft(padded_sino, axis=0)
    freq_filtered = sino_fft * fourier_filter
    sino_filtered.buffer = np.real(ifft(freq_filtered, axis=0))

    # Resizing to original sinogram dimensions
    sino_filtered.buffer = sino_filtered.buffer[pad_width.shape[0]:pad_width.shape[0] + sinogram.buffer.shape[0], :]
    sino_filtered.height, sino_filtered.width = sino_filtered.get_buffer().shape[0], sino_filtered.get_buffer().shape[1]

    return sino_filtered


def ramlak_filter(sinogram, detector_spacing):
    projection_size_padded = helper.next_power_of_two(sinogram.buffer.shape[0])
    pad_width = np.zeros([(projection_size_padded - sinogram.buffer.shape[0]) // 2, sinogram.buffer.shape[1]])
    padded_sino = np.vstack((pad_width, sinogram.buffer))
    padded_sino = np.vstack((padded_sino, pad_width))

    constantFactor = -1.0 / (float(np.power(np.pi, 2) * np.power(detector_spacing, 2)))
    n1 = np.arange(1, int(projection_size_padded / 2) + 1, 2)
    n2 = np.arange(int(projection_size_padded / 2) - 1, 0, -2)
    n = np.concatenate((n1, n2))
    filter_array = np.zeros(projection_size_padded)
    filter_array[0] = 1 / (4 * np.power(detector_spacing, 2))
    filter_array[1::2] = constantFactor / np.power(n, 2)

    # Ram Lak convolver initialized in spatial domain. Computing it's FT with sinogram instead
    # of convolving in spatial domain
    fourier_filter = 2 * np.real(fft(filter_array)).reshape(-1, 1)

    # Optional : plotting the filter kernel in spatial domain
    plt.plot(filter_array)
    plt.show()
    # Optional : plotting the filter kernel in freq domain
    plt.plot(fourier_filter)
    plt.show()

    # Apply ramp filter in Fourier domain. Filtering row wise (along each detector line)
    sino_filtered = grid.Grid(padded_sino.shape[0], padded_sino.shape[1],
                              (sinogram.get_spacing()[0], sinogram.get_spacing()[1]))
    sino_fft = fft(padded_sino, axis=0)
    freq_filtered = sino_fft * fourier_filter
    sino_filtered.buffer = np.real(ifft(freq_filtered, axis=0))

    # Resizing to original sinogram dimensions
    sino_filtered.buffer = sino_filtered.buffer[pad_width.shape[0]:pad_width.shape[0] + sinogram.buffer.shape[0], :]
    sino_filtered.height, sino_filtered.width = sino_filtered.get_buffer().shape[0], sino_filtered.get_buffer().shape[1]

    return sino_filtered


def backproject(sinogram, recon_size_x, recon_size_y, spacing):
    recon_img = grid.Grid(recon_size_x, recon_size_y, spacing)  # spacing should be received as a tuple
    detector_length = sinogram.get_size()[1]
    deltaS = sinogram.get_spacing()[1]
    projections = len(sinogram.get_buffer()[0])

    theta = np.linspace(0, 180, projections, endpoint=False)

    # Using library for to create FBP for comparison
    img_fbp = iradon(sinogram.get_buffer(), theta=theta, output_size=recon_size_x, filter_name=None, circle=False)

    for x in range(recon_img.get_size()[0]):
        for y in range(recon_img.get_size()[1]):
            w = recon_img.index_to_physical(x, y)  # not needed then
            for i in range(0, len(theta)):  # last loop
                angle = (theta[i]) * (np.pi / 180)
                s = w[0] * (recon_img.get_spacing()[0] * np.cos(angle)) + w[1] * (
                        recon_img.get_spacing()[0] * np.sin(angle))

                # compute detector element index from world coordinates
                s += detector_length / 2
                s /= deltaS

                if sinogram.get_size()[0] >= s + 1 and s > 0:
                    val = sinogram.get_at_index(int(np.floor(s)), i)
                    recon_img.buffer[x][y] += val

    return recon_img.get_buffer(), img_fbp
