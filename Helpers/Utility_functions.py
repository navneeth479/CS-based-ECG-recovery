import numpy as np
import flat_panel_project_utils as utils


def line_integral(phantom, s, theta):
    N = phantom.get_size()[0]
    points = int(np.ceil(2 * np.sqrt(2) * N))
    t = np.linspace(-np.sqrt(2), np.sqrt(2), points)
    x = s * np.cos(theta) + t * np.sin(theta)
    y = -s * np.sin(theta) + t * np.cos(theta)
    cols = np.floor(x * N / 2) + N / 2
    rows = np.floor(y * N / 2) + N / 2
    line_vector = np.zeros(points, dtype=object)
    ray_sum = 0

    for i in range(points):
        # ray_sum += phantom.get_at_physical(x[i], y[i])
        line_vector[i] = (rows[i], cols[i])
        lv_set = list(set(line_vector))

    for xy_line_pair in lv_set:
        if 0 <= xy_line_pair[0] < N and 0 <= xy_line_pair[1] < N:
            # ray_sum += phantom.get_at_physical(xy_line_pair[0], xy_line_pair[1])
            ray_sum += utils.interpolate(phantom, xy_line_pair[0], xy_line_pair[1])
    return ray_sum


def next_power_of_two(value):
    value = value - 1
    return int(2 * pow(2, np.ceil(np.log(value) / np.log(2))))
