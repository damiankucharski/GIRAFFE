from pathlib import Path
from typing import Callable, Iterable, List, Sequence, Type, Union

import numpy as np

import giraffe.lib_types as lib_types
from giraffe.backend.backend import Backend
from giraffe.callback import Callback
from giraffe.crossover import crossover, tournament_selection_indexes
from giraffe.fitness import average_precision_fitness
from giraffe.globals import BACKEND as B
from giraffe.globals import DEVICE
from giraffe.mutation import get_allowed_mutations
from giraffe.node import OperatorNode
from giraffe.operators import MAX, MEAN, MIN, WEIGHTED_MEAN
from giraffe.population import initialize_individuals
from giraffe.tree import Tree
from giraffe.utils import first_uniques_mask


class Giraffe:
    """
    Main class for evolutionary model ensemble optimization.

    Giraffe uses genetic programming to evolve tree-based ensembles of machine learning models.
    The algorithm creates a population of trees where each tree represents a different way of
    combining model predictions. Through evolution (crossover and mutation), it searches for
    optimal ensemble structures that maximize a fitness function.

    Each tree has ValueNodes that contain tensor predictions from individual models, and
    OperatorNodes that define how to combine these predictions (e.g., mean, min, max, weighted mean).
    The evolution process selects and combines high-performing trees to produce better ensembles.

    Attributes:
        population_size: Number of individuals in the population
        population_multiplier: Factor determining how many additional trees to generate in each iteration
        tournament_size: Number of trees to consider in tournament selection
        fitness_function: Function used to evaluate the fitness of each tree
        callbacks: Collection of callbacks for monitoring/modifying the evolution process
        allowed_ops: Operator node types allowed in tree construction
        train_tensors: Dictionary mapping model names to their prediction tensors
        gt_tensor: Ground truth tensor for comparison
        population: Current population of trees
        additional_population: Additional trees generated during evolution
    """

    def __init__(
        self,
        preds_source: Union[Path, str, Iterable[Path], Iterable[str]],
        gt_path: Union[Path, str],
        population_size: int,
        population_multiplier: int,
        tournament_size: int,
        fitness_function: Callable[[Tree, lib_types.Tensor], float] = average_precision_fitness,
        allowed_ops: Sequence[Type[OperatorNode]] = (MEAN, MIN, MAX, WEIGHTED_MEAN),
        callbacks: Iterable[Callback] = tuple(),
        backend: Union[Backend, None] = None,
        seed: int = 0,
    ):
        """
        Initialize the Giraffe evolutionary algorithm.

        Args:
            preds_source: Source of model predictions, can be a path to directory or iterable of paths
            gt_path: Path to ground truth data
            population_size: Size of the population to evolve
            population_multiplier: Factor determining how many additional trees to generate
            tournament_size: Number of trees to consider in tournament selection
            fitness_function: Function to evaluate fitness of trees
            allowed_ops: Sequence of operator node types that can be used in trees
            callbacks: Iterable of callback objects for monitoring/modifying evolution
            backend: Optional backend implementation for tensor operations
            seed: Random seed for reproducibility
        """
        if backend is not None:
            Backend.set_backend(backend)
        if seed is not None:
            np.random.seed(seed)

        self.population_size = population_size
        self.population_multiplier = population_multiplier
        self.tournament_size = tournament_size
        self.fitness_function = fitness_function
        self.callbacks = callbacks
        self.allowed_ops = allowed_ops

        self.train_tensors, self.gt_tensor = self._build_train_tensors(preds_source, gt_path)
        self._validate_input()

        # state
        self.should_stop = False

        self.population = self._initialize_population()
        self.additional_population: List[Tree] = []  # for potential callbacks

    def _call_hook(self, hook_name):
        """
        Call a specific hook on all registered callbacks.

        Args:
            hook_name: Name of the hook to call
        """
        for callback in self.callbacks:
            getattr(callback, hook_name)(self)

    def _initialize_population(self):
        """
        Initialize the population of trees.

        Creates simple trees using available prediction tensors.

        Returns:
            List of initialized Tree objects
        """
        return initialize_individuals(self.train_tensors, self.population_size)

    def _calculate_fitnesses(self, trees: None | List[Tree] = None):
        """
        Calculate fitness values for the given trees.

        Args:
            trees: List of trees to evaluate. If None, uses the current population.

        Returns:
            NumPy array of fitness values
        """
        if trees is None:
            trees = self.population
        return np.array([self.fitness_function(tree, self.gt_tensor) for tree in trees])

    def run_iteration(self):
        """
        Run a single iteration of the evolutionary algorithm.

        This method:
        1. Calculates fitness values for the current population
        2. Performs tournament selection and crossover to create new trees
        3. Applies mutations to some of the new trees
        4. Removes duplicate trees from the population
        """
        fitnesses = self._calculate_fitnesses(self.population)
        while len(self.additional_population) < (self.population_multiplier * self.population_size):
            idx1, idx2 = tournament_selection_indexes(fitnesses, self.tournament_size)
            parent_1, parent_2 = self.population[idx1], self.population[idx2]
            new_tree_1, new_tree_2 = crossover(parent_1, parent_2)
            self.additional_population += [new_tree_1, new_tree_2]

        models, ids = list(self.train_tensors.keys()), list(self.train_tensors.values())
        for tree in self.additional_population:
            if np.random.rand() > tree.mutation_chance:
                allowed_mutations = np.array(get_allowed_mutations(tree))
                chosen_mutation = np.random.choice(allowed_mutations)
                mutated_tree = chosen_mutation(
                    tree,
                    models=models,
                    ids=ids,
                    allowed_ops=self.allowed_ops,
                )
                self.additional_population.append(mutated_tree)
        joined_population = np.array(self.population + self.additional_population)
        codes = np.array([tree.__repr__() for tree in joined_population])
        mask = first_uniques_mask(codes)
        joined_population = joined_population[mask]

        # TODO: handle duplicates
        self.additional_population = []

    def train(self, iterations: int):
        """
        Run the evolutionary algorithm for a specified number of iterations.

        Args:
            iterations: Number of evolution iterations to run
        """
        self._call_hook("on_evolution_start")

        for _ in range(iterations):
            self._call_hook("on_generation_start")  # possibly move to run_iteration instead
            self.run_iteration()
            self._call_hook("on_generation_end")

            if self.should_stop:
                break

        self._call_hook("on_evolution_end")

    def _build_train_tensors(self, preds_source, gt_path):
        """
        Load prediction tensors and ground truth from files.

        Args:
            preds_source: Source of model predictions (path or iterable of paths)
            gt_path: Path to ground truth data

        Returns:
            Tuple of (train_tensors dictionary, ground truth tensor)
        """
        if isinstance(preds_source, str):
            preds_source = Path(preds_source)
        if isinstance(preds_source, Path):
            tensor_paths = list(preds_source.glob("*"))
        else:
            tensor_paths = preds_source

        train_tensors = {}
        for tensor_path in tensor_paths:
            train_tensors[Path(tensor_path).name] = B.load(tensor_path, DEVICE)

        gt_tensor = B.load(gt_path, DEVICE)
        return train_tensors, gt_tensor

    def _validate_input(self, fix_swapped=True):  # no way to change this argument for now TODO
        """
        Validate that all input tensors have compatible shapes.

        Checks if all prediction tensors have the same shape and if the ground truth
        tensor has a compatible shape. Can optionally fix swapped dimensions in the
        ground truth tensor.

        Args:
            fix_swapped: If True, attempts to fix swapped dimensions in ground truth tensor

        Raises:
            ValueError: If tensor shapes are incompatible and cannot be fixed
        """
        # check if all tensors have the same shape
        shapes = [B.shape(tensor) for tensor in self.train_tensors.values()]
        if len(set(shapes)) > 1:
            raise ValueError(f"Tensors have different shapes: {shapes}")

        if B.shape(self.gt_tensor) != shapes[0]:
            if fix_swapped:
                if (shapes[0] == B.shape(self.gt_tensor)[::-1]) and (len(shapes[0]) == 2):
                    self.gt_tensor = B.reshape(self.gt_tensor, shapes[0])

            else:
                raise ValueError(f"Ground truth tensor has different shape than input tensors: {shapes[0]} != {B.shape(self.gt_tensor)}")
