from giraffe.backend.type import TensorBackend
import numpy as np
from typing import Union, override


class NumpyBackend(TensorBackend):
    def __init__(self, tensor: Union[np.ndarray, list, tuple, int, float]):
        tensor = np.asarray(tensor)
        super().__init__(tensor)

    def mean(self, axis=None) -> TensorBackend:
        return NumpyBackend(np.mean(self.tensor, axis=axis))

    def max(self, axis=None) -> TensorBackend:
        return NumpyBackend(np.max(self.tensor, axis=axis))

    def min(self, axis=None) -> TensorBackend:
        return NumpyBackend(np.min(self.tensor, axis=axis))

    def sum(self, axis=None) -> TensorBackend:
        return NumpyBackend(np.sum(self.tensor, axis=axis))

    def numpy(self) -> np.ndarray:
        return self.tensor

    def clip(self, min, max) -> TensorBackend:
        return NumpyBackend(np.clip(self.tensor, min, max))

    def log(self) -> TensorBackend:
        return NumpyBackend(np.log(self.tensor))

    def float(self) -> TensorBackend:
        return NumpyBackend(self.tensor.astype(float))

    def shape(self) -> tuple[int]:
        return self.tensor.shape

    def reshape(self, *args, **kwargs) -> TensorBackend:
        return NumpyBackend(self.tensor.reshape(*args, **kwargs))

    def squeeze(self) -> TensorBackend:
        return NumpyBackend(np.squeeze(self.tensor))

    def unsqueeze(self, axis) -> TensorBackend:
        return NumpyBackend(np.expand_dims(self.tensor, axis))

    def get_tensortype(self):
        return np.ndarray
