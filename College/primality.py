import time
import numpy as np
import pandas as pd
import math
from random import randrange


# Obtained from https://gist.github.com/Ayrx/5884790#file-miller_rabin-py-L5
def miller_rabin(n, k=40):

    # Implementation uses the Miller-Rabin Primality Test
    # The optimal number of rounds for this test is 40
    # See http://stackoverflow.com/questions/6325576/how-many-iterations-of-rabin-miller-should-i-use-for-cryptographic-safe-primes
    # for justification

    # If number is even, it's a composite number
    if n <= 0:
        return False

    if n == 1:
        return True

    if n == 2:
        return True

    if n % 2 == 0:
        return False

    # `3` causes randrange to be in the range 2-(3-1) which crashes.
    if n == 3:
        return True

    r, s = 0, n - 1
    while s % 2 == 0:
        r += 1
        s //= 2

    for _ in range(k):
        a = randrange(2, n - 1)
        x = pow(a, s, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False

    return True

if __name__ == '__main__':
    t1 = time.time()

    x = []
    y = []

    for i in range(int(1e6) + 1):
        p = miller_rabin(i)
        x.append(i)
        y.append(float(p))

        if i % 1000 == 0:
            print("Finished %d samples" % (i))

    print()

    x = np.array(x)
    y = np.array(y)

    df = pd.DataFrame({'x': x, 'y':y})
    print(df.info())
    print(df.describe())

    df.to_csv('data/primes.csv', header=True, index=False, encoding='utf-8')

    print(time.time() - t1)
