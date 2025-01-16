from giraffe.globals import BACKEND as B
import numpy as np
from giraffe.node import OperatorNode, ValueNode, Node
from typing import Union


class Tree:
    def __init__(self, root: ValueNode, mutation_chance=0.1):
        self.root = root

        if isinstance(self.root, OperatorNode):
            raise Exception("Cannot get evaluation of tree with OpNode as root")

        self.nodes: Union[None, dict[str, list]] = None
        self.mutation_chance = mutation_chance
        self.update_nodes()

    def update_nodes(self):
        self.nodes = {"value_nodes": [], "op_nodes": []}
        for node in self.root.get_nodes():
            if isinstance(node, ValueNode):
                self.nodes["value_nodes"].append(node)
            else:
                self.nodes["op_nodes"].append(node)
