import numpy as np
from loguru import logger

from giraffe.tree import Tree


def tournament_selection_indexes(fitnesses: np.ndarray, tournament_size: int = 5) -> np.ndarray:
    """
    Selects parent indices for crossover using tournament selection.

    In tournament selection, a subset of individuals (of size tournament_size) is randomly
    selected from the population, and the one with the highest fitness is chosen as a parent.
    This process is repeated to select the second parent.

    Args:
        fitnesses: Array of fitness values for the entire population
        tournament_size: Number of individuals to include in each tournament

    Returns:
        Array with indices of the two selected parents

    Raises:
        ValueError: If tournament_size is too large relative to population size
    """
    assert len(fitnesses.shape) == 1
    if tournament_size >= (len(fitnesses) - 1):
        raise ValueError(f"Size of the tournament should be at least 1 less than number of participans but{len(fitnesses)=} and {tournament_size=}")

    if len(fitnesses) < (2 * tournament_size):
        logger.warning(
            f"Tournament size ({tournament_size}), is small related to the population size ({len(fitnesses)})."
            "The population should be at least twice as large as tournament for more stable parent selection"
        )

    candidates = np.random.choice(fitnesses, size=(2, tournament_size))
    selected = np.argmax(candidates, axis=1).ravel()
    assert selected.shape == (2,)

    return selected


def crossover(tree1: Tree, tree2: Tree, node_type=None):
    """
    Performs crossover between two parent trees to produce two offspring trees.

    Crossover works by selecting a random node from each parent tree and swapping
    the subtrees rooted at those nodes. This creates two new offspring trees that
    contain genetic material from both parents.

    Args:
        tree1: First parent tree
        tree2: Second parent tree
        node_type: Type of nodes to consider for crossover points ('value_nodes' or 'op_nodes').
                   If None, a random suitable type will be chosen.

    Returns:
        Tuple of two new Tree objects created by crossover

    Raises:
        ValueError: If node_type is 'op_nodes' but one or both trees don't have operator nodes
    """
    if node_type is None:
        allowable_node_types = ["value_nodes"]  # TODO: this may be worth refactoring along with "get_random_node" to not use string but types instead

        if (len(tree1.nodes["op_nodes"]) > 0) & (len(tree2.nodes["op_nodes"]) > 0):
            allowable_node_types.append("op_nodes")
        nodes_type = np.random.choice(allowable_node_types)
    else:
        if node_type == "op_nodes" and not ((len(tree1.nodes["op_nodes"]) > 0) & (len(tree2.nodes["op_nodes"]) > 0)):
            raise ValueError("Node type was chosen to be operator nodes but there are not operator nodes in at least one of the parents")
        nodes_type = node_type

    tree1, tree2 = tree1.copy(), tree2.copy()
    node1, node2 = tree1.get_random_node(nodes_type), tree2.get_random_node(nodes_type)
    branch1, branch2 = node1.copy_subtree(), node2.copy_subtree()

    tree1.replace_at(node1, branch2).recalculate()
    tree2.replace_at(node2, branch1).recalculate()

    return tree1, tree2
