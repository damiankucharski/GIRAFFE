import numpy as np

from giraffe.backend.backend_interface import BackendInterface


class NumpyBackend(BackendInterface):
    @staticmethod
    def tensor(x):
        return np.asarray(x)

    @staticmethod
    def concat(tensors, axis=0):
        if len(tensors[0].shape) == 1:
            tensors = [t[np.newaxis, :] for t in tensors]
        return np.concatenate(tensors, axis)

    @staticmethod
    def mean(x, axis=None):
        return np.mean(x, axis)

    @staticmethod
    def max(x, axis=None):
        return np.max(x, axis)

    @staticmethod
    def min(x, axis=None):
        return np.min(x, axis)

    @staticmethod
    def sum(x, axis=None):
        return np.sum(x, axis)

    @staticmethod
    def to_numpy(x):
        return x

    @staticmethod
    def clip(x, min, max):
        return np.clip(x, min, max)

    @staticmethod
    def log(x):
        return np.log(x)

    @staticmethod
    def to_float(x):
        return x.astype(float)

    @staticmethod
    def shape(x):
        return x.shape

    @staticmethod
    def reshape(x, *args, **kwargs):
        return np.reshape(x, *args, **kwargs)

    @staticmethod
    def squeeze(x):
        return np.squeeze(x)

    @staticmethod
    def unsqueeze(x, axis):
        return np.expand_dims(x, axis)
