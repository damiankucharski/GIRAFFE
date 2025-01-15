from giraffe.backend.type import TensorBackend
from giraffe.backend.numpy_backend import NumpyBackend
import pytest
import numpy as np

BACKENDS = [NumpyBackend]


@pytest.mark.parametrize(
    "array, expected_mean, axis, expected_shape",
    [
        ([1, 2, 3, 4], 2.5, None, ()),
        ([[1, 2], [3, 4]], [2.0, 3.0], 0, (2,)),
        ([[1, 2], [3, 4]], [1.5, 3.5], 1, (2,)),
    ],
)
def test_mean(array, expected_mean, axis, expected_shape):
    for backend in BACKENDS:
        tensor = backend(array)
        result = tensor.mean(axis=axis).numpy()
        np.testing.assert_array_equal(result, expected_mean)
        np.testing.assert_array_equal(result.shape, expected_shape)


@pytest.mark.parametrize(
    "array, expected_sum, axis, expected_shape",
    [
        ([1, 2, 3, 4], 10, None, ()),
        ([[1, 2], [3, 4]], [4, 6], 0, (2,)),
        ([[1, 2], [3, 4]], [3, 7], 1, (2,)),
    ],
)
def test_sum(array, expected_sum, axis, expected_shape):
    for backend in BACKENDS:
        tensor = backend(array)
        result = tensor.sum(axis=axis).numpy()
        np.testing.assert_array_equal(result, expected_sum)
        np.testing.assert_array_equal(result.shape, expected_shape)


@pytest.mark.parametrize(
    "array, expected_max, axis, expected_shape",
    [
        ([1, 2, 3, 4], 4, None, ()),
        ([[1, 2], [3, 4]], [3, 4], 0, (2,)),
        ([[1, 2], [3, 4]], [2, 4], 1, (2,)),
    ],
)
def test_max(array, expected_max, axis, expected_shape):
    for backend in BACKENDS:
        tensor = backend(array)
        result = tensor.max(axis=axis).numpy()
        np.testing.assert_array_equal(result, expected_max)
        np.testing.assert_array_equal(result.shape, expected_shape)


@pytest.mark.parametrize(
    "array, expected_min, axis, expected_shape",
    [
        ([1, 2, 3, 4], 1, None, ()),
        ([[1, 2], [3, 4]], [1, 2], 0, (2,)),
        ([[1, 2], [3, 4]], [1, 3], 1, (2,)),
    ],
)
def test_min(array, expected_min, axis, expected_shape):
    for backend in BACKENDS:
        tensor = backend(array)
        result = tensor.min(axis=axis).numpy()
        np.testing.assert_array_equal(result, expected_min)
        np.testing.assert_array_equal(result.shape, expected_shape)


@pytest.mark.parametrize(
    "array, min_val, max_val, expected_result",
    [
        ([1, 2, 3, 4], 2, 3, [2, 2, 3, 3]),
        ([[1, 2], [3, 4]], 2, 3, [[2, 2], [3, 3]]),
    ],
)
def test_clip(array, min_val, max_val, expected_result):
    for backend in BACKENDS:
        tensor = backend(array)
        result = tensor.clip(min_val, max_val).numpy()
        np.testing.assert_array_equal(result, expected_result)


@pytest.mark.parametrize(
    "array, new_shape, expected_shape",
    [
        ([1, 2, 3, 4], (2, 2), (2, 2)),
        ([[1, 2], [3, 4]], (4,), (4,)),
    ],
)
def test_reshape(array, new_shape, expected_shape):
    for backend in BACKENDS:
        tensor = backend(array)
        result = tensor.reshape(new_shape).numpy()
        np.testing.assert_array_equal(result.shape, expected_shape)


@pytest.mark.parametrize(
    "array, expected_result",
    [
        ([[1], [2], [3], [4]], [1, 2, 3, 4]),
        ([[[1, 2]], [[3, 4]]], [[1, 2], [3, 4]]),
    ],
)
def test_squeeze(array, expected_result):
    for backend in BACKENDS:
        tensor = backend(array)
        result = tensor.squeeze().numpy()
        np.testing.assert_array_equal(result, expected_result)


@pytest.mark.parametrize(
    "array, axis, expected_shape",
    [
        ([1, 2, 3, 4], 0, (1, 4)),
        ([[1, 2], [3, 4]], 1, (2, 1, 2)),
    ],
)
def test_unsqueeze(array, axis, expected_shape):
    for backend in BACKENDS:
        tensor = backend(array)
        result = tensor.unsqueeze(axis).numpy()
        np.testing.assert_array_equal(result.shape, expected_shape)
