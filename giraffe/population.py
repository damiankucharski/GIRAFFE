from typing import Dict, List

import numpy as np
from loguru import logger

from giraffe.lib_types import Tensor
from giraffe.node import ValueNode
from giraffe.tree import Tree


def initialize_individuals(tensors_dict: Dict[str, Tensor], n: int, exclude_ids=tuple()) -> List[Tree]:
    """
    Initialize a population of individuals (trees) from a dictionary of tensors.

    This function creates simple trees, each with a root node containing a different tensor
    from the provided dictionary. The tensors are selected randomly from the dictionary.

    Args:
        tensors_dict: Dictionary mapping model IDs to their tensor representations
        n: Number of individuals (trees) to create
        exclude_ids: Optional tuple of model IDs to exclude from selection

    Returns:
        List of initialized Tree objects

    Raises:
        Exception: If n is greater than the number of available tensors after exclusions
    """
    logger.info(f"Initializing {n} individuals")
    logger.debug(f"Available tensors: {len(tensors_dict)}, excluded IDs: {len(exclude_ids)}")

    order = np.arange(len(tensors_dict))
    np.random.shuffle(order)
    logger.trace("Shuffled tensor order")

    ids_list = list(tensors_dict.keys())
    tensors_list = list(tensors_dict.values())

    new_trees = []
    count = 0
    for idx in order:
        _id = ids_list[idx]
        tensor = tensors_list[idx]
        if count >= n:
            break
        if _id in exclude_ids:
            logger.trace(f"Skipping excluded ID: {_id}")
            continue

        logger.debug(f"Creating tree with tensor ID: {_id}")
        root: ValueNode = ValueNode(children=None, value=tensor, id=_id)
        tree = Tree.create_tree_from_root(root)
        new_trees.append(tree)
        count += 1

    if count < n:
        logger.error(f"Could not generate enough individuals. Requested: {n}, generated: {count}")
        raise Exception("Could not generate as many examples")

    logger.info(f"Successfully initialized {len(new_trees)} individuals")
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
    logger.debug(f"Selecting {n} best trees from population of {len(trees)}")

    # Sort indices by fitness in descending order
    sorted_indices = np.argsort(-fitnesses)
    logger.trace(f"Sorted fitness indices: {sorted_indices[: min(5, len(sorted_indices))]}")

    # Select the top n indices
    selected_indices = sorted_indices[:n]

    # Return selected trees and their fitnesses
    selected_trees = [trees[i] for i in selected_indices]
    selected_fitnesses = fitnesses[selected_indices]

    logger.debug(f"Selected {len(selected_trees)} trees with fitness range: {selected_fitnesses.min():.4f} - {selected_fitnesses.max():.4f}")
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
    logger.debug(f"Selecting up to {n} Pareto-optimal trees from population of {len(trees)}")
    from giraffe.pareto import maximize, minimize, paretoset

    # Create a 2D array with [fitness, nodes_count] for each tree
    objectives_array = np.zeros((len(trees), 2), dtype=float)
    for i, (tree, fitness) in enumerate(zip(trees, fitnesses, strict=True)):
        objectives_array[i, 0] = fitness  # Maximize fitness
        objectives_array[i, 1] = tree.nodes_count  # Minimize nodes count

    logger.trace(f"Created objectives array with shape {objectives_array.shape}")

    # Get Pareto-optimal mask using maximize for fitness and minimize for nodes count
    pareto_mask = paretoset(objectives_array, [maximize, minimize])
    pareto_count = np.sum(pareto_mask)
    logger.debug(f"Found {pareto_count} Pareto-optimal trees")

    # Get indices of Pareto-optimal trees
    pareto_indices = np.where(pareto_mask)[0]

    # If we have more Pareto-optimal trees than n, select the n with highest fitness
    if len(pareto_indices) > n:
        logger.debug(f"Too many Pareto-optimal trees ({len(pareto_indices)}), selecting top {n} by fitness")
        # Sort by fitness (descending)
        sorted_indices = pareto_indices[np.argsort(-fitnesses[pareto_indices])]
        selected_indices = sorted_indices[:n]
    else:
        selected_indices = pareto_indices
        logger.debug(f"Using all {len(selected_indices)} Pareto-optimal trees")

    # Return selected trees and their fitnesses
    selected_trees = [trees[i] for i in selected_indices]
    selected_fitnesses = fitnesses[selected_indices]

    if len(selected_trees) > 0:
        logger.debug(f"Selected {len(selected_trees)} trees with fitness range: {selected_fitnesses.min():.4f} - {selected_fitnesses.max():.4f}")
    else:
        logger.warning("No trees selected in Pareto optimization")

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
    logger.info(f"Selecting {n} trees using Pareto-then-sorted strategy")

    # Get all Pareto-optimal trees without limiting the number
    # Internal implementation of choose_pareto uses a limit, so we use a large number
    # to effectively get all Pareto trees
    all_pareto_trees, all_pareto_fitnesses = choose_pareto(trees, fitnesses, len(trees))
    logger.debug(f"Found {len(all_pareto_trees)} Pareto-optimal trees")

    # If we have more Pareto-optimal trees than n, select the n with highest fitness
    if len(all_pareto_trees) > n:
        logger.debug(f"Too many Pareto trees ({len(all_pareto_trees)}), selecting top {n}")
        return choose_n_best(all_pareto_trees, all_pareto_fitnesses, n)

    # If we have exactly n Pareto trees, return them
    if len(all_pareto_trees) == n:
        logger.debug(f"Exactly {n} Pareto trees, returning all of them")
        return all_pareto_trees, all_pareto_fitnesses

    # We need to fill the remainder with sorted trees
    remaining_slots = n - len(all_pareto_trees)
    logger.debug(f"Need {remaining_slots} more trees to reach target of {n}")

    # Create a list of non-Pareto trees by excluding Pareto trees
    pareto_trees_set = set(all_pareto_trees)
    non_pareto_trees = []
    non_pareto_fitnesses = []

    for i, tree in enumerate(trees):
        if tree not in pareto_trees_set:
            non_pareto_trees.append(tree)
            non_pareto_fitnesses.append(fitnesses[i])

    logger.debug(f"Found {len(non_pareto_trees)} non-Pareto trees")
    non_pareto_fitnesses_np = np.array(non_pareto_fitnesses)

    # Use choose_n_best to select the remaining trees
    best_remaining_trees, best_remaining_fitnesses = choose_n_best(non_pareto_trees, non_pareto_fitnesses_np, remaining_slots)
    logger.debug(f"Selected {len(best_remaining_trees)} additional trees by fitness")

    # Combine Pareto and sorted selections
    selected_trees = all_pareto_trees + best_remaining_trees
    selected_fitnesses = np.concatenate([all_pareto_fitnesses, best_remaining_fitnesses])

    logger.info(f"Total selection: {len(selected_trees)} trees ({len(all_pareto_trees)} Pareto + {len(best_remaining_trees)} by fitness)")
    return selected_trees, selected_fitnesses
