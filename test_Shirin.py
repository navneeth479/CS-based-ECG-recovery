import Grid as g
import phantom as p
#import SheppLoganShow as shpshow
#import matplotlib.pyplot as plt

shepp_phantom = p.phantom(n = 256, p_type = 'Modified Shepp-Logan', ellipses = None)

grid = g.Grid(512, 512, 1)

origin_grid = grid.get_origin()

phantom_grid = grid.set_buffer(shepp_phantom)

grid.get_buffer()


#print("get at physical", g.get_at_physical(grid, 200, 150))
