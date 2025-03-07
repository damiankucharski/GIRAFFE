from typing import Tuple, cast, Self

import numpy as np
from loguru import logger

from giraffe.globals import BACKEND as B
from giraffe.globals import DEVICE
from giraffe.node import Node, OperatorNode, ValueNode, check_if_both_types_same_node_variant
from giraffe.utils import Pickle


class Tree:
    def __init__(self, root: ValueNode, mutation_chance=0.1):
        self.root = root

        if isinstance(self.root, OperatorNode):
            raise Exception("Cannot get evaluation of tree with OpNode as root")

        self.nodes: dict[str, list] = {"value_nodes": [], "op_nodes": []}
        self.mutation_chance = mutation_chance
        self.update_nodes()

    def update_nodes(self):
        self.nodes = {"value_nodes": [], "op_nodes": []}
        root_nodes = self.root.get_nodes()
        for node in root_nodes:
            if isinstance(node, ValueNode):
                self.nodes["value_nodes"].append(node)
            else:
                self.nodes["op_nodes"].append(node)

    @staticmethod
    def create_tree_from_root(root: ValueNode, mutation_chance=0.1):
        tree = Tree(root, mutation_chance)
        return tree

    @property
    def evaluation(self):
        # WARNING: This may not make sense for cases other than binary classification (Squeezing)
        # return B.squeeze(self.root.evaluation if self.root.evaluation is not None else self.root.calculate())
        return self.root.evaluation if self.root.evaluation is not None else self.root.calculate()

    @property
    def nodes_count(self):
        return len(self.nodes["value_nodes"]) + len(self.nodes["op_nodes"])

    def _clean_evals(self):
        for node in self.nodes["value_nodes"]:
            node.evaluation = None

    def recalculate(self):
        self._clean_evals()
        return self.evaluation

    def copy(self):
        root_copy: ValueNode = cast(ValueNode, self.root.copy_subtree())
        return Tree.create_tree_from_root(root_copy)

    def prune_at(self, node: Node):  # remove node from the tree along with its children
        if node not in self.nodes["value_nodes"] and node not in self.nodes["op_nodes"]:
            raise ValueError("Node not found in tree")

        if node.parent is None:
            raise ValueError("Cannot prune root node")
        if isinstance(node.parent, OperatorNode) and (
            len(node.parent.children) < 2
        ):  # if only child of op node is to be pruned, remove the parent too
            return self.prune_at(node.parent)

        subtree_nodes = node.get_nodes()

        for subtree_node in subtree_nodes:
            if isinstance(subtree_node, ValueNode):
                self.nodes["value_nodes"].remove(subtree_node)
            else:
                self.nodes["op_nodes"].remove(subtree_node)

        node.parent.remove_child(node)

        self._clean_evals()
        return node

    def append_after(self, node: Node, new_node: Node):
        if node not in self.nodes["value_nodes"] and node not in self.nodes["op_nodes"]:
            raise ValueError("Node not found in tree")

        if check_if_both_types_same_node_variant(type(node), type(new_node)):
            raise ValueError("Cannot append node of the same type")

        subtree_nodes = new_node.get_nodes()

        for subtree_node in subtree_nodes:
            if isinstance(subtree_node, ValueNode):
                self.nodes["value_nodes"].append(subtree_node)
            else:
                self.nodes["op_nodes"].append(subtree_node)

        node.add_child(new_node)
        self._clean_evals()

    def replace_at(
        self, at: Node, replacement: Node
    ) -> Self:  # like prune at and then append after parent, but without parameters adjustment (may be worth it to reimplement)
        assert isinstance(replacement, type(at)), "Replacement must be of the same type as the node being replaced"
        at_parent = at.parent

        if at_parent is None:
            assert isinstance(self.root, ValueNode), "Root must be a value node"
            assert isinstance(replacement, ValueNode), "Replacement for root must be a value node"
            logger.warning("Node at replacement is root node")
            self.root = replacement
        else:
            at_parent.replace_child(at, replacement)

        if isinstance(at, ValueNode):
            self.nodes["value_nodes"].remove(at)
            self.nodes["value_nodes"].append(replacement)
        else:
            self.nodes["op_nodes"].remove(at)
            self.nodes["op_nodes"].append(replacement)

        self._clean_evals()
        return self

    def get_random_node(self, nodes_type: str | None = None, allow_root=True, allow_leaves=True):
        if self.root.children == []:
            if allow_root:
                if nodes_type is None or nodes_type == "value_nodes":
                    return self.root
                else:
                    raise ValueError("Tree has only root node and nodes_type is not value_nodes")
            else:
                raise ValueError("Tree has only root node and allow_root is set to False")

        if nodes_type is None:
            nodes_type = np.random.choice(["value_nodes", "op_nodes"])

        assert nodes_type is not None, "Nodes type cannot be None"

        order = np.arange(len(self.nodes[nodes_type]))
        for i in order:
            node = self.nodes[nodes_type][i]
            if (allow_leaves or node.children != []) and (allow_root or node != self.root):
                return node
        raise ValueError("No node found that complies to the constraints")

    def get_unique_value_node_ids(self):
        return list(set([node.id for node in self.nodes["value_nodes"]]))

    def save_tree_architecture(self, output_path):  # TODO: needs adjustment for weighted node
        copy_tree = self.copy()
        for value_node in copy_tree.nodes["value_nodes"]:
            value_node.value = value_node.evaluation = None

        Pickle.save(output_path, copy_tree)

    @staticmethod
    def load_tree_architecture(architecture_path) -> "Tree":  # TODO: needs adjustmed for weighted node
        return Pickle.load(architecture_path)

    @staticmethod
    def load_tree(architecture_path, preds_directory, tensors={}) -> Tuple["Tree", dict]:
        current_tensors = {}
        current_tensors.update(tensors)  # tensors argument is mutable and we do not want to modify it

        loaded = Tree.load_tree_architecture(architecture_path)
        for value_node in loaded.nodes["value_nodes"]:
            node_id = value_node.id
            if node_id not in current_tensors:
                current_tensors[node_id] = B.load(preds_directory / str(node_id), DEVICE)
            value_node.value = current_tensors[node_id]

        return loaded, current_tensors

    def __repr__(self):
        return "_".join(node.code for node in self.root.get_nodes())
