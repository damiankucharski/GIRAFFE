import numpy as np

from giraffe.globals import BACKEND as B
from giraffe.lib_types import Tensor
from giraffe.tree import Tree


def average_precision_fitness(tree: Tree, gt: Tensor) -> float:
    """
    Calculate the Average Precision (AP) score as a fitness measure.

    Average Precision summarizes a precision-recall curve as the weighted mean of precisions
    achieved at each threshold, with the increase in recall from the previous threshold used
    as the weight. This metric is particularly useful for binary classification problems
    with imbalanced classes.

    Args:
        tree: The tree whose evaluation will be compared against ground truth
        gt: Ground truth tensor containing binary labels

    Returns:
        Average Precision score as a float between 0 and 1 (higher is better)
    """
    from sklearn.metrics import average_precision_score

    pred_np: np.ndarray = B.to_numpy(tree.evaluation)
    gt_np: np.ndarray = B.to_numpy(B.to_float(gt))
    return average_precision_score(gt_np, pred_np)  # type: ignore


def roc_auc_score_fitness(tree: Tree, gt: Tensor) -> float:
    """
    Calculate the Area Under the ROC Curve (AUC-ROC) score as a fitness measure.

    The AUC-ROC score represents the probability that a randomly chosen positive instance
    is ranked higher than a randomly chosen negative instance. While it's a popular metric
    for classification tasks, it can give optimistic results on highly imbalanced datasets.

    Args:
        tree: The tree whose evaluation will be compared against ground truth
        gt: Ground truth tensor containing binary labels

    Returns:
        ROC AUC score as a float between 0 and 1 (higher is better)
    """
    from sklearn.metrics import roc_auc_score

    pred_np: np.ndarray = B.to_numpy(tree.evaluation)
    gt_np: np.ndarray = B.to_numpy(B.to_float(gt))
    return roc_auc_score(gt_np, pred_np)  # type: ignore
