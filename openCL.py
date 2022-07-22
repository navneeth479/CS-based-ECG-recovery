import datetime
import numpy as np
import pyopencl as cl
import Grid as grid
import flat_panel_project_utils as utils



a_np = utils.circle2D(256, 256)
utils.show(a_np, "phantom 1")

b_np = utils.dotsgrid2D(256, 256)
utils.show(b_np, "phantom 2")


t0 = datetime.datetime.now()
res = np.zeros_like(a_np)
for i in range(a_np.shape[0]):
    for j in range(a_np.shape[1]):
        res[i, j] = np.add(a_np[i, j], b_np[i, j])

t = datetime.datetime.now() - t0
print("Time taken by nested loop", t.total_seconds())
utils.show(res, "nested loop result")
size = a_np.shape[0]

platform = cl.get_platforms()
GPU = platform[0].get_devices()

ctx = cl.Context(GPU)
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)
res_np = np.zeros_like(a_np)
res_g = cl.Buffer(ctx, mf.WRITE_ONLY, res_np.nbytes)

prg = cl.Program(ctx, """
    __kernel void sum(
         __global float *a_g, __global float *b_g, __global float *res_g, int size_test) {
            int i = get_global_id(1);
            int j = get_global_id(0);
            int size = get_global_size(1);
            res_g[i + size * j] = 0;
            res_g[i + size * j] =  a_g[i + size * j] + b_g[i + size * j] ;
    }
""").build()

t1 = datetime.datetime.now()

prg.sum(queue, a_np.shape, None, a_g, b_g, res_g, np.int32(size))

cl.enqueue_copy(queue, res_np, res_g)

t2 = datetime.datetime.now() - t1

# Check on CPU with Numpy:

print("Time taken", t2.total_seconds())
utils.show(res_np, "GPU result")


val = list(np.array(res_np == res).reshape(-1, ))
print(f'{val.count(True)}/{len(val)}')
