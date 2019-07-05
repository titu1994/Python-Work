import time

import numba
import numpy as np


@numba.jit(nopython=True, cache=True)
def e(n: int):
    x = 3.0

    if n <= 0:
        return x

    result = 0.0
    for i in range(n, 0, -1):
        num = -i
        denom = (i + 3) + result
        result = (num / denom)

    x = x + result
    return x


def abs_diff(n: int):
    for ni in range(1, n + 1):
        val = e(ni)
        print("Value of Euler's constant after %d iterations" % (ni), val)

        diff = np.abs((np.e - val))
        print("Absolute difference : ", diff)
        print()


if __name__ == '__main__':
    # iters = 15
    # val = e(iters)
    # print("Value of Euler's constant after %d iterations" % (iters), val)
    #
    # diff = np.abs((np.e - val))
    # print("Absolute difference : ", diff)

    abs_diff(15)

    # time
    num_tests = 10000
    t1 = time.time()
    for i in range(num_tests):
        e(20)
    t2 = time.time()

    print("Time for %d runs = " % (num_tests), ((t2 - t1)))
    print("Time per run = ", ((t2 - t1) / float(num_tests)))