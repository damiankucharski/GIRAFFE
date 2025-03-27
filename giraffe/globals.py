import os
from typing import Type

from giraffe.backend.backend import Backend
from giraffe.backend.backend_interface import BackendInterface

# Device to use for tensor operations, can be set via environment variable
DEVICE = os.environ.get("DEVICE", None)


# ---- Postprocessing functions ----
def _passthrough(x):
    """
    Default postprocessing function that simply returns the input unchanged.

    Args:
        x: Input to pass through

    Returns:
        The unchanged input
    """
    return x


# Global postprocessing function that will be applied to tree evaluations
postprocessing_function = _passthrough


def set_postprocessing_function(func):
    """
    Set the global postprocessing function.

    Args:
        func: The function to use for postprocessing tree evaluations
    """
    global postprocessing_function
    postprocessing_function = func


# ---- Backend configuration ----
# Initialize the backend based on environment variable or default to numpy
Backend.set_backend(os.environ.get("BACKEND", "numpy"))
BACKEND: Type[BackendInterface] = Backend.get_backend()


def set_backend(backend_name):
    """
    Set the tensor backend to use.

    Args:
        backend_name: Name of the backend to use ('numpy' or 'pytorch')
    """
    Backend.set_backend(backend_name)


def get_backend():
    """
    Get the current tensor backend.

    Returns:
        The current backend interface class
    """
    return Backend.get_backend()


# ----
