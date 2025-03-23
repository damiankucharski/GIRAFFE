import os
from typing import Type

from giraffe.backend.backend import Backend
from giraffe.backend.backend_interface import BackendInterface

DEVICE = os.environ.get("DEVICE", None)

# ----
def _passthrough(x):
    return x

postprocessing_function = _passthrough

def set_postprocessing_function(func):
    global postprocessing_function
    postprocessing_function = func
# ----
Backend.set_backend(os.environ.get("BACKEND", "numpy"))
BACKEND: Type[BackendInterface] = Backend.get_backend()

def set_backend(backend_name):
    Backend.set_backend(backend_name)

def get_backend():
    return Backend.get_backend()
# ----
