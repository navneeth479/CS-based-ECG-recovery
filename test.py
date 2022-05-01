import Grid as grid
import flat_panel_project_utils as utils
phantom = utils.shepp_logan(512)


grid_test = grid.Grid(512, 512, 2)
origin = grid_test.get_origin()
buffer = grid_test.buffer
buffer = phantom
print(buffer[10][120])

utils.show(phantom, "phantom")



