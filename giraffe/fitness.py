import numpy as np

from giraffe.globals import BACKEND as B
from giraffe.lib_types import Tensor
from giraffe.tree import Tree


def average_precision_fitness(tree: Tree, gt: Tensor) -> float:
    from sklearn.metrics import average_precision_score

    pred_np: np.ndarray = B.to_numpy(tree.evaluation)
    gt_np: np.ndarray = B.to_numpy(B.to_float(gt))
    return average_precision_score(gt_np, pred_np) # type: ignore


def roc_auc_score_fitness(tree: Tree, gt: Tensor) -> float:
    from sklearn.metrics import roc_auc_score

    pred_np: np.ndarray = B.to_numpy(tree.evaluation)
    gt_np: np.ndarray = B.to_numpy(B.to_float(gt))
    return roc_auc_score(gt_np, pred_np) # type: ignore
