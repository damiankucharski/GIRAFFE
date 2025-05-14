import numpy as np
import pytest

# Import the functions to test
from giraffe.pareto import maximize, minimize, paretoset


def test_simple_maximize():
    """Test with single maximize objective"""
    points = np.array([[1], [2], [3]])
    result = paretoset(points, [maximize])
    assert result == [False, False, True]


def test_simple_minimize():
    """Test with single minimize objective"""
    points = np.array([[1], [2], [3]])
    result = paretoset(points, [minimize])
    assert result == [True, False, False]


def test_two_objectives_maximize():
    """Test with two maximize objectives"""
    points = np.array([[1, 1], [2, 2], [3, 1], [1, 3]])
    result = paretoset(points, [maximize, maximize])
    assert result == [False, True, True, True]


def test_mixed_objectives():
    """Test with mix of maximize and minimize objectives"""
    points = np.array(
        [
            [1, 4],  # Dominated by all others
            [2, 3],  # Dominated by [3,2] and [4,1]
            [3, 2],  # Dominated by [4,1]
            [4, 1],  # Not dominated by any point
        ]
    )
    result = paretoset(points, [maximize, minimize])
    assert result == [False, False, False, True]


def test_mixed_objectives_multiple_optimal():
    """Test with mix of maximize and minimize objectives where multiple points are optimal"""
    points = np.array(
        [
            [4, 1],  # optimal, highest maximize
            [3, 0],  # best minimize
            [3, 3],  # dominated
            [4, 1],  # the same as first
        ]
    )
    result = paretoset(points, [maximize, minimize])
    assert result == [True, True, False, True]


def test_identical_points():
    """Test handling of identical points"""
    points = np.array([[1, 1], [1, 1], [2, 2]])
    result = paretoset(points, [maximize, maximize])
    assert result == [False, False, True]


def test_empty_array():
    """Test with empty array"""
    points = np.array([]).reshape(0, 2)
    result = paretoset(points, [maximize, maximize])
    assert len(result) == 0


def test_single_point():
    """Test with single point"""
    points = np.array([[1, 1]])
    result = paretoset(points, [maximize, maximize])
    assert result == [True]


def test_array_dimension_error():
    """Test error handling for incorrect array dimensions"""
    points = np.array([1, 2, 3])  # 1D array
    with pytest.raises(AssertionError):
        paretoset(points, [maximize])


def test_objective_count_mismatch():
    """Test error handling for mismatched number of objectives"""
    points = np.array([[1, 2], [3, 4]])
    with pytest.raises(AssertionError):
        paretoset(points, [maximize])  # Only one objective for 2D points


def test_three_objectives():
    """Test with three objectives"""
    points = np.array([[1, 1, 1], [2, 2, 2], [3, 1, 2], [1, 3, 2]])
    result = paretoset(points, [maximize, maximize, maximize])
    assert result == [False, True, True, True]


def test_all_dominated():
    """Test case where all points except one are dominated"""
    points = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
    result = paretoset(points, [maximize, maximize])
    assert result == [False, False, False, True]


def test_none_dominated():
    """Test case where no points are dominated"""
    points = np.array([[4, 1], [3, 2], [2, 3], [1, 4]])
    result = paretoset(points, [maximize, maximize])
    assert result == [True, True, True, True]
