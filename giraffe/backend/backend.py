from typing import Type

from giraffe.backend.backend_interface import BackendInterface
from giraffe.backend.numpy_backend import NumpyBackend


class Backend:
    _current_backend: Type[BackendInterface] = NumpyBackend

    @classmethod
    def set_backend(cls, backend_name):  # TODO: Add option to set backend by providing class instead
        if backend_name == "torch":
            from giraffe.backend.pytorch import PyTorchBackend

            cls._current_backend = PyTorchBackend
        elif backend_name == "numpy":
            cls._current_backend = NumpyBackend
        else:
            raise ValueError(f"Invalid backend: {backend_name}")

    @classmethod
    def get_backend(cls) -> Type[BackendInterface]:
        return cls._current_backend
