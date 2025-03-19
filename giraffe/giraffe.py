from pathlib import Path
from typing import Iterable, Union

from giraffe.backend.backend import Backend
from giraffe.globals import BACKEND as B
from giraffe.globals import DEVICE


class Giraffe:
    def __init__(
        self,
        preds_source: Union[Path, str, Iterable[Path], Iterable[str]],
        gt_path: Union[Path, str],
        backend: Union[Backend, None] = None,
    ):
        if backend is not None:
            Backend.set_backend(backend)

        self.train_tensors, self.gt_tensor = self._build_train_tensors(preds_source, gt_path)

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
