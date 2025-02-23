from typing import Sequence, Type

import numpy as np

from giraffe.node import MeanNode, OperatorNode, ValueNode
from giraffe.tree import Tree
from giraffe.types import Tensor


def append_new_node_mutation(
    tree: Tree, models: Sequence[Tensor], ids: None | Sequence[str | int] = None, allowed_ops: tuple[Type[OperatorNode], ...] = (MeanNode,)
):
    tree = tree.copy()

    if ids is None:
        ids = list(range(len(models)))
    else:
        assert len(models) == len(ids)

    idx_model = np.random.choice(ids)
    node = tree.get_random_node()

    val_node: ValueNode = ValueNode([], models[idx_model], ids[idx_model])
    if isinstance(node, ValueNode):
        random_op: Type[OperatorNode] = np.random.choice(np.asarray(allowed_ops))
        op_node: OperatorNode = random_op.create_node([])
        op_node.add_child(val_node)
        tree.append_after(node, op_node)
    else:
        tree.append_after(node, val_node)

    return tree
