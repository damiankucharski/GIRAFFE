from typing import Dict, List

import numpy as np

from giraffe.lib_types import Tensor
from giraffe.node import ValueNode
from giraffe.tree import Tree


def initialize_individuals(tensors_dict: Dict[str, Tensor], n: int, exclude_ids=tuple()) -> List[Tree]:
    order = np.arange(len(tensors_dict))
    np.random.shuffle(order)

    ids = np.array(list(tensors_dict.keys()))[order]
    tensors = np.array(list(tensors_dict.values()))[order]

    new_trees = []
    count = 0
    for tensor, _id in zip(tensors, ids, strict=False):
        if count >= n:
            break
        if _id in exclude_ids:
            continue
        root: ValueNode = ValueNode(children=None, value=tensor, id=_id)
        tree = Tree.create_tree_from_root(root)
        new_trees.append(tree)
        count += 1
    if count < n:
        raise Exception("Could not generate as many examples")

    return new_trees


def choose_n_best(trees: List[Tree], fitnesses: np.ndarray, n: int):
    """
    Select n trees with the highest fitness values.

    Args:
        trees: List of Tree objects
        fitnesses: Array of fitness values for each tree
        n: Number of trees to select

    Returns:
        List of selected trees and their corresponding fitness values
    """
    # Sort indices by fitness in descending order
    sorted_indices = np.argsort(-fitnesses)

    # Select the top n indices
    selected_indices = sorted_indices[:n]

    # Return selected trees and their fitnesses
    selected_trees = [trees[i] for i in selected_indices]
    selected_fitnesses = fitnesses[selected_indices]

    return selected_trees, selected_fitnesses


def choose_pareto(trees: List[Tree], fitnesses: np.ndarray, n: int):
    """
    Select up to n trees based on Pareto optimality.
    Optimizes for:
    - Maximizing fitness
    - Minimizing number of nodes in the tree

    Args:
        trees: List of Tree objects
        fitnesses: Array of fitness values for each tree
        n: Maximum number of trees to select

    Returns:
        List of selected trees and their corresponding fitness values
    """
    from giraffe.pareto import maximize, minimize, paretoset

    # Create a 2D array with [fitness, nodes_count] for each tree
    objectives_array = np.zeros((len(trees), 2), dtype=float)
    for i, (tree, fitness) in enumerate(zip(trees, fitnesses, strict=True)):
        objectives_array[i, 0] = fitness  # Maximize fitness
        objectives_array[i, 1] = tree.nodes_count  # Minimize nodes count

    # Get Pareto-optimal mask using maximize for fitness and minimize for nodes count
    pareto_mask = paretoset(objectives_array, [maximize, minimize])

    # Get indices of Pareto-optimal trees
    pareto_indices = np.where(pareto_mask)[0]

    # If we have more Pareto-optimal trees than n, select the n with highest fitness
    if len(pareto_indices) > n:
        # Sort by fitness (descending)
        sorted_indices = pareto_indices[np.argsort(-fitnesses[pareto_indices])]
        selected_indices = sorted_indices[:n]
    else:
        selected_indices = pareto_indices

    # Return selected trees and their fitnesses
    selected_trees = [trees[i] for i in selected_indices]
    selected_fitnesses = fitnesses[selected_indices]

    return selected_trees, selected_fitnesses


def choose_pareto_then_sorted(trees: List[Tree], fitnesses: np.ndarray, n: int):
    """
    First select Pareto-optimal trees, then fill the remainder (up to n) with
    the best sorted trees not already in the Pareto set.

    Args:
        trees: List of Tree objects
        fitnesses: Array of fitness values for each tree
        n: Total number of trees to select

    Returns:
        List of selected trees and their corresponding fitness values
    """
    # Get all Pareto-optimal trees without limiting the number
    # Internal implementation of choose_pareto uses a limit, so we use a large number
    # to effectively get all Pareto trees
    all_pareto_trees, all_pareto_fitnesses = choose_pareto(trees, fitnesses, len(trees))

    # If we have more Pareto-optimal trees than n, select the n with highest fitness
    if len(all_pareto_trees) > n:
        return choose_n_best(all_pareto_trees, all_pareto_fitnesses, n)

    # If we have exactly n Pareto trees, return them
    if len(all_pareto_trees) == n:
        return all_pareto_trees, all_pareto_fitnesses

    # We need to fill the remainder with sorted trees
    remaining_slots = n - len(all_pareto_trees)

    # Create a list of non-Pareto trees by excluding Pareto trees
    pareto_trees_set = set(all_pareto_trees)
    non_pareto_trees = []
    non_pareto_fitnesses = []

    for i, tree in enumerate(trees):
        if tree not in pareto_trees_set:
            non_pareto_trees.append(tree)
            non_pareto_fitnesses.append(fitnesses[i])

    non_pareto_fitnesses_np = np.array(non_pareto_fitnesses)

    # Use choose_n_best to select the remaining trees
    best_remaining_trees, best_remaining_fitnesses = choose_n_best(non_pareto_trees, non_pareto_fitnesses_np, remaining_slots)

    # Combine Pareto and sorted selections
    selected_trees = all_pareto_trees + best_remaining_trees
    selected_fitnesses = np.concatenate([all_pareto_fitnesses, best_remaining_fitnesses])

    return selected_trees, selected_fitnesses
