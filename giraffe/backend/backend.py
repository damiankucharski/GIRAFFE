from giraffe.backend.backend_interface import BackendInterface
from typing import Type


class Backend:
    _current_backend: Type[BackendInterface] | None = None

    @classmethod
    def set_backend(cls, backend_name):  # TODO: Add option to set backend by providing class instead
        if backend_name == "torch":
            from giraffe.backend.pytorch import PyTorchBackend

            cls._current_backend = PyTorchBackend
        elif backend_name == "numpy":
            from giraffe.backend.numpy_backend import NumpyBackend

            cls._current_backend = NumpyBackend
        else:
            raise ValueError(f"Invalid backend: {backend_name}")

    @classmethod
    def get_backend(cls):
        if cls._current_backend is None:
            cls.set_backend("numpy")  # Set a default
        return cls._current_backend
