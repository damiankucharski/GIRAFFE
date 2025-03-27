from typing import Callable, Sequence, Type

import numpy as np

from giraffe.lib_types import Tensor
from giraffe.node import MeanNode, OperatorNode, ValueNode
from giraffe.tree import Tree


def append_new_node_mutation(
    tree: Tree, models: Sequence[Tensor], ids: None | Sequence[str | int] = None, allowed_ops: tuple[Type[OperatorNode], ...] = (MeanNode,), **kwargs
):
    """
    Mutation that adds a new node to the tree.

    This mutation randomly selects an existing node in the tree and appends a new node as its
    child. If the selected node is a ValueNode, a new OperatorNode is created as an intermediary,
    and the new ValueNode is added as its child. If the selected node is an OperatorNode,
    a new ValueNode is directly appended to it.

    Args:
        tree: The tree to mutate
        models: Sequence of tensor models that can be used as values for the new ValueNode
        ids: Optional sequence of identifiers for the models. If None, indices will be used
        allowed_ops: Tuple of OperatorNode types that can be used when creating a new operator node
        **kwargs: Additional keyword arguments (ignored)

    Returns:
        A new Tree with the mutation applied
    """
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
    """
    Mutation that removes a branch from the tree.

    This mutation randomly selects a non-root, non-leaf node in the tree and removes it along
    with all its descendants, effectively pruning that branch from the tree.

    Args:
        tree: The tree to mutate
        **kwargs: Additional keyword arguments (ignored)

    Returns:
        A new Tree with the mutation applied

    Raises:
        AssertionError: If the tree has fewer than 3 nodes
    """
    tree = tree.copy()
    assert tree.nodes_count >= 3, "Tree is too small"
    node = tree.get_random_node(allow_leaves=False, allow_root=False)
    tree.prune_at(node)
    return tree


def new_tree_from_branch_mutation(tree: Tree, **kwargs):
    """
    Mutation that creates a new tree from a branch of the existing tree.

    This mutation randomly selects a non-root ValueNode, removes it from the tree along with
    its descendants, and creates a new tree with the removed node as its root.

    Args:
        tree: The tree to mutate
        **kwargs: Additional keyword arguments (ignored)

    Returns:
        A new Tree created from the selected branch

    Raises:
        AssertionError: If the tree has only one ValueNode
    """
    tree = tree.copy()
    assert len(tree.nodes["value_nodes"]) > 1
    node = tree.get_random_node(nodes_type="value_nodes", allow_leaves=True, allow_root=False)
    sapling_node = tree.prune_at(node)
    assert isinstance(sapling_node, ValueNode)
    new_tree = Tree.create_tree_from_root(sapling_node)
    return new_tree


def get_allowed_mutations(tree):
    """
    Determines which mutation operations are valid for a given tree.

    This function checks the tree's structure and size to determine which mutations
    can be safely applied without violating constraints.

    Args:
        tree: The tree to analyze

    Returns:
        A list of mutation functions that are valid for the given tree
    """
    allowed_mutations: list[Callable] = [
        append_new_node_mutation,
    ]

    if tree.nodes_count >= 3:
        allowed_mutations.append(lose_branch_mutation)
    if len(tree.nodes["value_nodes"]) > 1:
        allowed_mutations.append(new_tree_from_branch_mutation)
    return allowed_mutations
