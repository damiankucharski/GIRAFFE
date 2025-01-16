import os
from giraffe.backend.backend import Backend

Backend.set_backend(os.environ.get("BACKEND", "numpy"))
BACKEND = Backend.get_backend()


def set_backend(backend_name):
    Backend.set_backend(backend_name)


def get_backend():
    return Backend.get_backend()
