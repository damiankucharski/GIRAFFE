from abc import ABC, abstractmethod
import numpy as np
from typing import Any


class TensorBackend(ABC):
    def __init__(self, tensor):
        self.tensor = tensor

    def __getattribute__(self, name):
        try:
            return super().__getattribute__(name)
        except AttributeError:
            try:
                out = self.tensor.__getattribute__(name)
                if isinstance(out, self.get_tensortype()):
                    return self.__class__(out)
            except AttributeError as err:
                raise AttributeError(f"TensorBackend object has no attribute {name}") from err

    @abstractmethod
    def mean(self, axis=None) -> "TensorBackend":
        raise NotImplementedError()

    @abstractmethod
    def max(self, axis=None) -> "TensorBackend":
        raise NotImplementedError()

    @abstractmethod
    def min(self, axis=None) -> "TensorBackend":
        raise NotImplementedError()

    @abstractmethod
    def sum(self, axis=None) -> "TensorBackend":
        raise NotImplementedError()

    @abstractmethod
    def numpy(self) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def clip(self, min, max) -> "TensorBackend":
        raise NotImplementedError()

    @abstractmethod
    def log(self) -> "TensorBackend":
        raise NotImplementedError()

    @abstractmethod
    def float(self) -> "TensorBackend":
        raise NotImplementedError()

    @abstractmethod
    def shape(self) -> tuple[int]:
        raise NotImplementedError()

    @abstractmethod
    def reshape(self, *args, **kwargs) -> "TensorBackend":
        raise NotImplementedError()

    @abstractmethod
    def squeeze(self) -> "TensorBackend":
        raise NotImplementedError()

    @abstractmethod
    def unsqueeze(self, axis) -> "TensorBackend":
        raise NotImplementedError()

    @abstractmethod
    def get_tensortype(self) -> Any:
        raise NotImplementedError()
