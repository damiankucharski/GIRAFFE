from typing import Dict, Tuple
from giraffe.lib_types import Tensor

import numpy as np

from giraffe.node import ValueNode
from giraffe.tree import Tree


def initialize_individuals(tensors_dict: Dict[str, Tensor], n:int, exclude_ids=tuple()):
    order = np.arange(len(tensors_dict))
    np.random.shuffle(order)

    ids = np.array(list(tensors_dict.keys()))[order]
    tensors = np.array(list(tensors_dict.values()))[order]

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
        count += 1
    if count < n:
        raise Exception("Could not generate as many examples")

    return new_trees
