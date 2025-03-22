from pathlib import Path
from typing import Callable, Iterable, List, Union, Sequence, Type

import numpy as np

import giraffe.lib_types as lib_types
from giraffe.backend.backend import Backend
from giraffe.callback import Callback
from giraffe.crossover import crossover, tournament_selection_indexes
from giraffe.fitness import average_precision_fitness
from giraffe.globals import BACKEND as B
from giraffe.globals import DEVICE
from giraffe.population import initialize_individuals
from giraffe.mutation import get_allowed_mutations
from giraffe.tree import Tree
from giraffe.node import OperatorNode
from giraffe.operators import MIN, MAX, WEIGHTED_MEAN, MEAN
from giraffe.utils import first_uniques_mask


class Giraffe:
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
        self.additional_population: List[Tree] = [] # for potential callbacks

    def _call_hook(self, hook_name):
        for callback in self.callbacks:
            getattr(callback, hook_name)(self)

    def _initialize_population(self):
       return initialize_individuals(self.train_tensors, self.population_size)

    def _calculate_fitnesses(self, trees: None | List[Tree] = None):
        if trees is None:
            trees = self.population
        return np.array([self.fitness_function(tree, self.gt_tensor) for tree in trees])

    def run_iteration(self):
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
        self._call_hook("on_evolution_start")

        for _ in range(iterations):
            self._call_hook("on_generation_start") # possibly move to run_iteration instead
            self.run_iteration()
            self._call_hook("on_generation_end")

            if self.should_stop:
                break

        self._call_hook("on_evolution_end")

    def _build_train_tensors(self, preds_source, gt_path):
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
