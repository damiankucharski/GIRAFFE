import numpy as np
from loguru import logger

from giraffe.tree import Tree


def tournament_selection_indexes(fitnesses: np.ndarray, tournament_size: int = 5) -> np.ndarray:
    assert len(fitnesses.shape) == 1
    if len(fitnesses) >= (tournament_size - 1):
        raise ValueError(
            "Size of the tournament should be at least 1 less than number of participans but" f"{len(fitnesses)=} and {tournament_size=}"
        )

    if len(fitnesses) < (2 * tournament_size):
        logger.warning(
            f"Tournament size ({tournament_size}), is small related to the population size ({len(fitnesses)})."
            "The population should be at least twice as large as tournament for more stable parent selection"
        )

    candidates = np.random.choice(fitnesses, size=(2, tournament_size))
    selected = np.argmax(candidates, axis=1).ravel()
    assert selected.shape == (2,)

    return selected


def crossover(tree1: Tree, tree2: Tree):
    allowable_node_types = ["value_nodes"]  # TODO: this may be worth refactoring along with "get_random_node" to not use string but types instead

    if (len(tree1.nodes["operator_nodes"]) > 0) & (len(tree2.nodes["operator_nodes"]) > 0):
        allowable_node_types.append("operator_nodes")
