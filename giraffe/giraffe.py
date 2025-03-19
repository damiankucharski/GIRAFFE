from pathlib import Path
from typing import Callable, Iterable, Union

import numpy as np

from giraffe.backend.backend import Backend
from giraffe.callback import Callback
from giraffe.globals import BACKEND as B
from giraffe.globals import DEVICE
from giraffe.tree import Tree
from giraffe.types import Tensor


class Giraffe:
    def __init__(
        self,
        preds_source: Union[Path, str, Iterable[Path], Iterable[str]],
        gt_path: Union[Path, str],
        population_size: int,
        population_multiplier: int,
        tournament_size: int,
        fitness_function: Callable[[Tree, Tensor], float],
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

        self.train_tensors, self.gt_tensor = self._build_train_tensors(preds_source, gt_path)
        self._validate_input()


    def _call_hook(self, hook_name):
        for callback in self.callbacks:
            getattr(callback, hook_name)(self)


    def run_iteration(self):
        pass

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
