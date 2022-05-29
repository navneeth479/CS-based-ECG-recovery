import Grid as grid
from skimage.transform import iradon
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import Helpers.Utility_functions as helper
from scipy.fftpack import fft, ifft, fftfreq


def create_sinogram(phantom, projections, detector_spacing, detector_size, angular_scan_range):
    # improvement...grid phantom not needed. Just pass grid class instance.
    grid_phantom = grid.Grid(len(phantom), len(phantom[0]), (1, 1))
    grid_phantom.set_buffer(phantom)
    angular_step = angular_scan_range / projections
    detector_length = detector_spacing * detector_size
    maxSIndex = int(detector_length / detector_spacing)

    N = grid_phantom.get_size()
    points = int(np.ceil(2 * np.sqrt(2) * N[0]))
    theta = np.linspace(0, angular_scan_range, projections)
    num_of_Angles = len(theta)
    s = np.linspace(-np.sqrt(2), np.sqrt(2), points)
    grid_sino = grid.Grid(maxSIndex, num_of_Angles, (angular_step, detector_spacing))
    grid_sino.set_origin(0, maxSIndex / 2)
    for i in range(points):
        for j in range(num_of_Angles):
            grid_sino.set_at_index(i, j, helper.line_integral(grid_phantom, s[i], theta[j]))
    return grid_sino


def ramp_filter(sinogram, detector_spacing):
    # Zero padding calculation and calculating ramp filter kernel
    # projection_size_padded_1 = \
    #              max(64, int(2 ** np.ceil(np.log2(2 * sinogram.get_size()[0]))))
    projection_size_padded = helper.next_power_of_two(sinogram.get_size()[0])
    pad_width = ((0, 0), (0, projection_size_padded - sinogram.get_size()[0]))
    padded_sino = np.pad(sinogram.get_buffer(), pad_width, mode='constant', constant_values=0)
    delta_f = 1 / (detector_spacing * projection_size_padded)
    f = fftfreq(projection_size_padded, d=delta_f)
    fourier_filter = 0.5 * np.abs(f)

    # Optional : plotting the filter kernel in freq domain
    plt.plot(fourier_filter)
    plt.show()

    # Apply ramp filter in Fourier domain. Filtering row wise (along each detector line)
    sino_filtered = grid.Grid(padded_sino.shape[0], padded_sino.shape[1],
                              (sinogram.get_spacing()[0], sinogram.get_spacing()[1]))
    for i in range(padded_sino.shape[0]):
        sino_fft = fft(padded_sino[i, :])
        freq_filtered = sino_fft * fourier_filter
        sino_filtered.buffer[i, :] = np.real(ifft(freq_filtered))

    # Resizing to original sinogram dimensions
    sino_filtered.buffer = sino_filtered.get_buffer()[:, :sinogram.get_size()[0]]
    sino_filtered.height, sino_filtered.width = sino_filtered.get_buffer().shape[0], sino_filtered.get_buffer().shape[1]

    return sino_filtered


def ramlak_filter(sinogram, detector_spacing):
    projection_size_padded = helper.next_power_of_two(sinogram.get_size()[0])
    pad_width = ((0, 0), (0, projection_size_padded - sinogram.get_size()[0]))
    padded_sino = np.pad(sinogram.get_buffer(), pad_width, mode='constant', constant_values=0)

    constantFactor = -1.0 / (float(np.power(np.pi, 2) * np.power(detector_spacing, 2)))
    n1 = np.arange(1, int(projection_size_padded / 2) + 1, 2)
    n2 = np.arange(int(projection_size_padded / 2) - 1, 0, -2)
    n = np.concatenate((n1, n2))
    filter_array = np.zeros(projection_size_padded)
    filter_array[0] = 1 / (4 * np.power(detector_spacing, 2))
    filter_array[1::2] = constantFactor / np.power(n, 2)

    # Ram Lak convolver initialized in spatial domain. Computing it's FT with sinogram instead
    # of convolving in spatial domain
    fourier_filter = 2 * np.real(fft(filter_array))

    # Optional : plotting the filter kernel in spatial domain
    plt.plot(filter_array)
    plt.show()
    # Optional : plotting the filter kernel in freq domain
    plt.plot(fourier_filter)
    plt.show()

    # Apply ramp filter in Fourier domain. Filtering row wise (along each detector line)
    sino_filtered = grid.Grid(padded_sino.shape[0], padded_sino.shape[1],
                              (sinogram.get_spacing()[0], sinogram.get_spacing()[1]))
    for i in range(padded_sino.shape[0]):
        sino_fft = fft(padded_sino[i, :])
        freq_filtered = sino_fft * fourier_filter
        sino_filtered.buffer[i, :] = np.real(ifft(freq_filtered))

    # Resizing to original sinogram dimensions
    sino_filtered.buffer = sino_filtered.get_buffer()[:, :sinogram.get_size()[0]]

    return sino_filtered


def backproject(sinogram, recon_size_x, recon_size_y, spacing):
    recon_img = grid.Grid(recon_size_x, recon_size_y, spacing)  # spacing should be received as a tuple
    maxSIndex = sinogram.get_size()[1]
    deltaS = sinogram.get_spacing()[1]
    projections = sinogram.get_size()[0]
    deltaTheta = sinogram.get_spacing()[0]
    detector_length = maxSIndex * deltaS

    theta = np.linspace(0, 180, projections, endpoint=False)
    img_fbp = iradon(sinogram.get_buffer(), theta=theta, circle=True)

    return img_fbp
