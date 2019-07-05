import time
import numba
import numpy as np


@numba.jit(nopython=True, cache=True)
def pi(n: int):
    x = 3.0

    if n <= 0:
        return x

    result = 0.0
    denom_outer = 5 + (n - 1) * 2

    for i in range(n, 0, -1):
        num = i * (i + 2)
        denom = denom_outer + result
        result = (num / denom)
        denom_outer -= 2

    x = x + result  # = 4 / (pi - 2)
    x = (4. / x) + 2.
    return x


def abs_diff(n: int):
    for ni in range(1, n + 1):
        val = pi(ni)
        print("Value of Pi after %d iterations" % (ni), val)

        diff = np.abs((np.pi - val))
        print("Absolute difference : ", diff)
        print()


if __name__ == '__main__':
    # iters = 15
    # val = pi(iters)
    # print("Value of Euler's constant after %d iterations" % (iters), val)
    # diff = np.abs((np.pi - val))
    # print("Absolute difference : ", diff)

    abs_diff(20)

    # time
    num_tests = 10000
    t1 = time.time()
    for i in range(num_tests):
        pi(20)
    t2 = time.time()

    print("Time for %d runs = " % (num_tests), ((t2 - t1)))
    print("Time per run = ", ((t2 - t1) / float(num_tests)))