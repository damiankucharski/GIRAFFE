import numpy as np

from giraffe.node import ValueNode
from giraffe.tree import Tree


def initialize_individuals(tensors: np.ndarray, ids: np.ndarray, n, exclude_ids=tuple()):
    assert len(tensors) == len(ids)
    order = np.arange(len(tensors))
    tensors, ids = tensors[order], ids[order]

    new_trees = []
    count = 0
    for tensor, _id in zip(tensors, ids, strict=False):
        if count >= n:
            break
        if _id in exclude_ids:
            continue
        root: ValueNode = ValueNode(children=None, value=tensor, id=_id)
        tree = Tree.create_tree_from_root(root)
        new_trees.append(tree)
    if count < n:
        raise Exception("Could not generate as many examples")

    return new_trees
