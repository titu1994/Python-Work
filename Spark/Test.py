from pyspark import SparkContext
import random
from time import time

if __name__ == "__main__":
    part = 2
    spark = SparkContext(master="local", appName="Pi Calculate")

    N = int(1e6)

    def calc_pi(p):
        x, y = random.random(), random.random()
        return 1 if ((x*x + y*y) < 1) else 0

    t1 = time()

    count = 0

    for i in range(N):
        count += calc_pi(i)

    t2 = time()

    print("Pi (Python) is roughly %f" % (4.0 * count / N))
    print("Time required : ", (t2 - t1))

    t1 = time()

    count = (
        spark.parallelize([0] * N, part)
             .flatMap(lambda x: [calc_pi(p) for p in range(int(N / part))])
             .reduce(lambda a, b: a + b)
    )

    t2 = time()

    print("Pi (Spark) is roughly %f" % (4.0 * count / N))
    print("Time required : ", (t2 - t1))

    spark.stop()
