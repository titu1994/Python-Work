import time
import numpy as np
from numba import cuda


@cuda.jit("void(float32[:, :], float32[:, :])")
def add_inplace(x, y):
    i, j = cuda.grid(2)  # get our global location on the cuda grid

    if i < x.shape[0] and j < x.shape[1]:  # check whether we exceed our bounds
        x[i][j] = x[i][j] + y[i][j]  # inplace _var_add into x

@cuda.jit("void(float32[:, :], float32[:, :], float32[:, :])")
def add_external(x, y, z):
    i, j = cuda.grid(2)  # get our global location on the cuda grid

    if i < x.shape[0] and j < x.shape[1]:  # check whether we exceed our bounds
        z[i][j] = x[i][j] + y[i][j]  # _var_add to a zero buffer


if __name__ == '__main__':
    NUM_TESTS = 25

    print("Num gpus : ", cuda.gpus)

    x = np.arange(int(1e8), dtype=np.float32).reshape((10000, 10000))
    y = np.arange(-int(1e8), int(1e8), step=2, dtype=np.float32).reshape((10000, 10000))

    print("x", x.shape, "y", y.shape)
    print()

    z_buffer = np.zeros_like(x, dtype=np.float32)
    x_copy = np.copy(x)

    threads_per_block = (32, 32)
    blocks_per_grid = ((int(x.shape[0] // threads_per_block[0])) + 1,
                       (int(x.shape[1] // threads_per_block[1])) + 1)

    print("Number of blocks : ", blocks_per_grid)
    print("Number of threads per block: ", threads_per_block)
    print()

    x = cuda.to_device(x)
    y = cuda.to_device(y)

    # _var_add inplace
    # pre compile
    add_inplace[blocks_per_grid, threads_per_block](x, y)

    t1 = time.time()
    for i in range(NUM_TESTS):
        add_inplace[blocks_per_grid, threads_per_block](x, y)

    x = x.copy_to_host()
    t2 = time.time()

    print("Time inplace : ", (t2 - t1) / float(NUM_TESTS))
    print()

    # _var_add external
    z_buffer = cuda.to_device(z_buffer)
    x_copy = cuda.to_device(x_copy)

    # pre compile
    add_external[blocks_per_grid, threads_per_block](x_copy, y, z_buffer)

    t1 = time.time()
    for i in range(NUM_TESTS):
        add_external[blocks_per_grid, threads_per_block](x_copy, y, z_buffer)

    z_buffer = z_buffer.copy_to_host()
    t2 = time.time()
    print("Time buffer : ", (t2 - t1) / float(NUM_TESTS))

    cuda.synchronize()
    cuda.close()