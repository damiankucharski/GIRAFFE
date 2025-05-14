# Backend API

This section documents the backend components of GIRAFFE, which handle tensor operations.

## Backend Interface

The abstract interface that all backend implementations must follow.

```python
from giraffe.backend.backend_interface import BackendInterface
```

::: giraffe.backend.backend_interface.BackendInterface
    options:
      show_source: true

## Backend Factory

Factory class for managing tensor backends.

```python
from giraffe.backend.backend import Backend
```

::: giraffe.backend.backend.Backend
    options:
      show_source: true

## Global Backend Configuration

Functions and variables for configuring the backend.

```python
from giraffe.globals import BACKEND, set_backend, get_backend, DEVICE
```

::: giraffe.globals
    options:
      members:
        - BACKEND
        - set_backend
        - get_backend
        - DEVICE
      show_source: true
