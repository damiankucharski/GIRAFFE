from typing import Callable, Sequence, Type

import numpy as np

from giraffe.lib_types import Tensor
from giraffe.node import MeanNode, OperatorNode, ValueNode
from giraffe.tree import Tree


def append_new_node_mutation(
    tree: Tree, models: Sequence[Tensor], ids: None | Sequence[str | int] = None, allowed_ops: tuple[Type[OperatorNode], ...] = (MeanNode,), **kwargs
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


def lose_branch_mutation(tree: Tree, **kwargs):
    tree = tree.copy()
    assert tree.nodes_count >= 3, "Tree is too small"
    node = tree.get_random_node(allow_leaves=False, allow_root=False)
    tree.prune_at(node)
    return tree


def new_tree_from_branch_mutation(tree: Tree, **kwargs):
    tree = tree.copy()
    assert len(tree.nodes["value_nodes"]) > 1
    node = tree.get_random_node(nodes_type="value_nodes", allow_leaves=True, allow_root=False)
    sapling_node = tree.prune_at(node)
    assert isinstance(sapling_node, ValueNode)
    new_tree = Tree.create_tree_from_root(sapling_node)
    return new_tree


def get_allowed_mutations(tree):
    allowed_mutations: list[Callable] = [
        append_new_node_mutation,
    ]

    if tree.nodes_count >= 3:
        allowed_mutations.append(lose_branch_mutation)
    if len(tree.nodes["value_nodes"]) > 1:
        allowed_mutations.append(new_tree_from_branch_mutation)
    return allowed_mutations
