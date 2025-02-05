import os
from typing import Type

from giraffe.backend.backend import Backend
from giraffe.backend.backend_interface import BackendInterface

Backend.set_backend(os.environ.get("BACKEND", "numpy"))
BACKEND: Type[BackendInterface] = Backend.get_backend()
DEVICE = os.environ.get("DEVICE", None)


def set_backend(backend_name):
    Backend.set_backend(backend_name)


def get_backend():
    return Backend.get_backend()
