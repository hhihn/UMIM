import tensorflow as tf
from tensorflow.python.ops.init_ops  import _compute_fans
import numpy as np
from tensorflow.keras.initializers import Initializer
from sklearn.cluster import KMeans


class InitCentersKMeans(Initializer):
    """ Initializer for initialization of centers of RBF network
        by clustering the given data set.
    # Arguments
        X: matrix, dataset
    """

    def __init__(self, X, max_iter=100):
        self.X = X
        self.max_iter = max_iter
        super().__init__()

    def __call__(self, shape, dtype=None):
        assert shape[1:] == self.X.shape[1:]

        n_centers = shape[0]
        km = KMeans(n_clusters=n_centers, max_iter=self.max_iter, verbose=0)
        km.fit(self.X)
        return km.cluster_centers_


class HeNormalExpertInitializer(tf.keras.initializers.Initializer):

    def __init__(self, numexp, mode="fan_in"):
        self.numexp = numexp
        self.scale = 2.0
        self.mode = mode

    def __call__(self, shape, dtype=None):
        if len(shape) != 3:
            raise ValueError("Invalid Input: Expected 3 dims, got %d" %len(shape))
        expert_shape = (shape[0], shape[1])
        scale = self.scale
        fan_in, fan_out = _compute_fans(shape)
        if self.mode == "fan_in":
            scale /= max(1., fan_in)
        elif self.mode == "fan_out":
            scale /= max(1., fan_out)
        else:
            scale /= max(1., (fan_in + fan_out) / 2.)
        stddev = np.sqrt(scale)
        weights = []
        for m in range(self.numexp):
            weights.append(tf.random.normal(expert_shape, stddev=stddev, dtype=dtype))
        weights = tf.stack(weights)
        weights = tf.reshape(weights, shape)
        return weights

    def get_config(self):  # To support serialization
        return {'numexp': self.numexp}


class HeUniformExpertInitializer(tf.keras.initializers.Initializer):

    def __init__(self, numexp):
        self.numexp = numexp
        self.scale = 2.0 * numexp

    def __call__(self, shape, dtype=None):
        if len(shape) < 3:
            raise ValueError("Invalid Input: Expected at least 3 dims, got %d" %len(shape))
        if len(shape) == 3:
            expert_shape = (shape[0], shape[1])
        if len(shape) == 5:
            expert_shape = (shape[0], shape[1], shape[2], shape[3])
        fan_in, fan_out = _compute_fans(shape)
        limit = tf.sqrt(self.scale / fan_in)
        weights = []
        for m in range(self.numexp):
            e_limit = limit
            weights.append(tf.random.uniform(expert_shape, minval=-e_limit, maxval=e_limit, dtype=dtype))
        weights = tf.stack(weights)
        weights = tf.reshape(weights, shape)
        return weights

    def get_config(self):  # To support serialization
        return {'numexp': self.numexp}
