from sklearn.metrics import average_precision_score, roc_auc_score

from giraffe.globals import BACKEND as B
from giraffe.tree import Tree
from giraffe.types import Tensor


def average_precision_fitness(tree: Tree, gt: Tensor):
    pred = B.to_numpy(tree.evaluation)
    gt = B.to_numpy(B.to_float(gt))
    return average_precision_score(gt, pred)


def roc_auc_score_fitness(tree: Tree, gt: Tensor):
    pred = B.to_numpy(tree.evaluation)
    gt = B.to_numpy(B.to_float(gt))
    return roc_auc_score(gt, pred)
