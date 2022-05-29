import pyconrad
import math
import numpy as np

pyconrad.setup_pyconrad()
pyconrad.start_gui()


def shepp_logan(xy):
    _ = pyconrad.ClassGetter('edu.stanford.rsl.tutorial.phantoms')
    return _.SheppLogan(xy).as_numpy()

def cencentric_circles(xy, disks):
    x = np.arange(-xy, xy, 1)
    y = np.arange(-xy, xy, 1)
    xx, yy = np.meshgrid(x, y, sparse=True)
    uniform_disk = (xx ** 2 + yy ** 2 < 40000)

    numdisks = disks
    concentric_disk = np.zeros((512, 512))
    disk_thickness = 20  # pixels
    for i in range(numdisks * 2 - 1):

        if i == 0:
            concentric_disk = (xx ** 2 + yy ** 2 < (disk_thickness * (i + 1)) ** 2)  #
        elif i % 2 and np.sqrt((disk_thickness * (i + 2)) ** 2) < 256:
            #         print(np.sqrt((disk_thickness * (i+2))**2))
            concentric_disk += ((xx ** 2 + yy ** 2 > (disk_thickness * (i + 1)) ** 2)) & (
                        xx ** 2 + yy ** 2 < (disk_thickness * (i + 2)) ** 2)

    if disks > 1:
        return concentric_disk
    else:
        return uniform_disk


def show(numpy_array, name):
    intermediate_grid = pyconrad.PyGrid.from_numpy(numpy_array)
    intermediate_grid.show(name)


def interpolate(grid, x, y):
    # Calculate the four surrounding data points in range [0,1]
    x_floor = math.floor(x)
    y_floor = math.floor(y)
    x_floor_plus_one = x_floor + 1
    y_floor_plus_one = y_floor + 1
    x_p = x - x_floor
    y_p = y - y_floor

    # Check boundary conditions
    if x_floor < 0 or x_floor > grid.get_size()[0] - 1:
        x_floor = None
    if x_floor_plus_one < 0 or x_floor_plus_one > grid.get_size()[0] - 1:
        x_floor_plus_one = None

    if y_floor < 0 or y_floor > grid.get_size()[1] - 1:
        y_floor = None
    if y_floor_plus_one < 0 or y_floor_plus_one > grid.get_size()[1] - 1:
        y_floor_plus_one = None
    # Get function values of the data points and setup function matrix
    a = grid.get_at_index(x_floor, y_floor) if (x_floor is not None and y_floor is not None) else 0.0
    b = grid.get_at_index(x_floor, y_floor_plus_one) if \
        (x_floor is not None and y_floor_plus_one is not None) else 0.0
    c = grid.get_at_index(x_floor_plus_one, y_floor) if \
        (x_floor_plus_one is not None and y_floor is not None) else 0.0
    d = grid.get_at_index(x_floor_plus_one, y_floor_plus_one) if \
        (x_floor_plus_one is not None and y_floor_plus_one is not None) else 0.0

    #                                       [f(0,0) f(0,1)] [1-y]
    # Calculate Interp value with    [1-x x] [f(1,0) f(1,1)] [ y ]
    val_matrix = np.array([[a, b], [c, d]])

    return np.matmul(np.matmul(np.array([1 - x_p, x_p]), val_matrix), np.transpose(np.array([1 - y_p, y_p])))
