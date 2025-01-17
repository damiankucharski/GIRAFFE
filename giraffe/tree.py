from giraffe.globals import BACKEND as B
import numpy as np
from giraffe.node import OperatorNode, ValueNode, Node
from typing import Union
from loguru import logger


class Tree:
    def __init__(self, root: ValueNode, mutation_chance=0.1):
        self.root = root

        if isinstance(self.root, OperatorNode):
            raise Exception("Cannot get evaluation of tree with OpNode as root")

        self.nodes: dict[str, list] = {"value_nodes": [], "op_nodes": []}
        self.mutation_chance = mutation_chance
        logger.debug("Updating nodes lists")
        logger.debug(f"Lists before update: {self.nodes=}")
        out = self.update_nodes()
        logger.debug(f"Lists after update: {self.nodes=}")

    def update_nodes(self):
        self.nodes = {"value_nodes": [], "op_nodes": []}
        root_nodes = self.root.get_nodes()
        for node in root_nodes:
            if isinstance(node, ValueNode):
                self.nodes["value_nodes"].append(node)
            else:
                self.nodes["op_nodes"].append(node)

        return 123

    @staticmethod
    def create_tree_from_root(root: ValueNode, mutation_chance=0.1):
        tree = Tree(root, mutation_chance)
        return tree

    @property
    def evaluation(self):
        return B.squeeze(self.root.evaluation if self.root.evaluation is not None else self.root.calculate())

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
        return Tree.create_tree_from_root(self.root.copy_subtree())

    def prune_at(self, node: Node):  # remove node from the tree along with its children
        if node.parent is None:
            raise Exception("Cannot prune root node")
        if isinstance(
            node.parent, OperatorNode
        ):  # and (len(node.parent.children) < 2): # I don't remember why one child only was allowed, commenting it out now.
            return self.prune_at(node.parent)

        subtree_nodes = node.get_nodes()

        for subtree_node in subtree_nodes:
            if isinstance(subtree_node, ValueNode):
                self.nodes["value_nodes"].remove(subtree_node)
            else:
                self.nodes["op_nodes"].remove(subtree_node)

        node.parent.remove_child(node)

        self._clean_evals()

    def append_after(self, node: Node, new_node: Node):
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
    ):  # like prune at and then append after parent, but without parameters adjustment (may be worth it to reimplement)
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

    def get_random_node(self, nodes_type: str | None = None, allow_root=True, allow_leaves=True):
        pass

    def get_unique_value_node_ids(self):
        pass

    def save_tree_architecture(self, output_path):  # TODO: needs adjustment for weighted node
        pass

    @staticmethod
    def load_tree_architecture(architecture_path):  # TODO: needs adjustmed for weighted node
        pass

    @staticmethod
    def load_tree(architecture_path, preds_directory, tensors={}):  # NEEDS TO USE BACKEND
        pass

    def __repr__(self):
        return "_".join(node.code for node in self.root.get_nodes())
