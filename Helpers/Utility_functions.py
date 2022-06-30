import numpy as np
from sympy import Point, Line, Segment


def line_integral(phantom, s, theta):
    diag = np.sqrt(np.power(phantom.get_size()[0] * phantom.get_spacing()[0], 2) + np.power(
        phantom.get_size()[1] * phantom.get_spacing()[0], 2))  # physical size spacing
    t = np.linspace(-diag / 2, diag / 2, int(diag))  # should be image diagonal
    x = s * np.cos(theta) - t * np.sin(theta)
    y = s * np.sin(theta) + t * np.cos(theta)
    ray_sum = 0

    for i in range(len(t)):
        ray_sum += phantom.get_at_physical(x[i], y[i])
    return ray_sum


def next_power_of_two(value):
    value = value - 1
    return int(2 * pow(2, np.ceil(np.log(value) / np.log(2))))

