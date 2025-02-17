from typing import Callable, Sequence

import numpy as np


def maximize(a, b):
    return a >= b


def minimize(a, b):
    return a <= b


def paretoset(array: np.ndarray, objectives: Sequence[Callable[[float, float], bool]]):
    assert len(array.shape) == 2, "Array should be one dimensional, where first dimension is number of points, second dimension number of objectives"

    n_points, n_objectives = array.shape

    assert len(objectives) == n_objectives

    domination_mask = [True for _ in range(n_points)]

    for i in range(n_points):  # checking if ith point should be on the pareto front
        for j in range(n_points):
            if i == j:
                continue
            if np.array_equal(array[i], array[j]):
                continue

            point_domination_mask = [f(array[j, k], array[i, k]) for k, f in enumerate(objectives)]
            if all(point_domination_mask):  # j dominates i because at least as good at all objectives
                domination_mask[i] = False
                break
    return domination_mask
