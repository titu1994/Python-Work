import tensorflow as tf
from tensorflow.contrib.eager.python import tfe
import numpy as np

from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

tf.enable_eager_execution()


def DisplayFractal(a):
    """Display an array of iteration counts as a
       colorful picture of a fractal."""
    a_cyclic = (6.28 * a / 20.0).reshape(list(a.shape) + [1])
    img = np.concatenate([10 + 20 * np.cos(a_cyclic),
                          30 + 50 * np.sin(a_cyclic),
                          155 - 80 * np.cos(a_cyclic)], 2)
    img[a == a.max()] = 0
    a = img
    a = np.uint8(np.clip(a, 0, 255))

    print(a.shape)
    plt.figure(dpi=300, figsize=(20, 20))
    plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    plt.imshow(a)
    #plt.show()
    plt.savefig('temp.png')


Y, X = np.mgrid[-1.3:1.3:0.001, -2:1:0.001]
Z = X + 1j * Y

num_gpus = tfe.num_gpus()

if num_gpus > 0:
    with tf.device('gpu:0'):
        xs = tf.constant(Z.astype(np.complex64))
        zs = tfe.Variable(xs)
        ns = tfe.Variable(tf.zeros_like(xs, tf.float32))
else:
    with tf.device('/cpu:0'):
        xs = tf.constant(Z.astype(np.complex64))
        zs = tfe.Variable(xs)
        ns = tfe.Variable(tf.zeros_like(xs, tf.float32))

# Operation to update the zs and the iteration count.
#
# Note: We keep computing zs after they diverge! This
#       is very wasteful! There are better, if a little
#       less simple, ways to do this.

def compute(zs, ns):
    for i in range(1000):
        # Compute the new values of z: z^2 + x
        zs_ = zs * zs + xs

        # Have we diverged with this new value?
        not_diverged = tf.abs(zs_) < 4

        zs = zs_
        ns = ns + tf.cast(not_diverged, tf.float32)
        
    return zs, ns

if num_gpus > 0:
    with tf.device('gpu:0'):
        zs, ns = compute(zs, ns)
else:
    with tf.device('/cpu:0'):
        zs, ns = compute(zs, ns)

DisplayFractal(ns.numpy())