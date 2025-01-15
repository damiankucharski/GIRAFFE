from typing import List, Optional, Self
from giraffe.backend.type import TensorBackend


class Node:
    """
    Nodes act as the fundamental building blocks of a tree,
    capable of holding children and a reference to their parent node.
    """

    def __init__(self, parent: Optional[Self] = None, children: Optional[List[Self]] = None):
        self.parent = parent
        self.children = children if children is not None else []
        self.type = None

    def add_child(self, child_node: Self):
        """
        Add a child to the Node.

        Parameters:
        - child_node: Node to be added as child
        """
        self.children.append(child_node)
        child_node.parent = self

    def remove_child(self, child_node: Self):
        self.children.remove(child_node)
        child_node.parent = None
        return child_node

    def replace_child(self, child, replacement_node):
        if replacement_node.parent is not None:
            raise ValueError("Replacement node already has a parent")
        self.add_child(replacement_node)
        self.remove_child(child)

    def get_nodes(self):
        """
        Get all nodes in the tree created by node and its subnodes.
        Returns:
        - List of all nodes in the tree in breadth-first order
        """
        nodes = [self]
        current_level = [self]

        while current_level:
            next_level = []
            for node in current_level:
                next_level.extend(node.children)
            nodes.extend(next_level)
            current_level = next_level

        return nodes

    def copy(self):
        """
        Create a copy of the node.
        It's children and parent references are not copied.

        Returns:
        - Copy of the node
        """
        raise NotImplementedError("Copy method not implemented")

    def copy_subtree(self):
        """
        Copy the subtree rooted at this node.

        Returns:
        - Copy of the subtree rooted at this node
        """
        self_copy = self.copy()
        self_copy.children = [child.copy_subtree() for child in self.children]
        for child in self_copy.children:
            child.parent = self_copy
        return self_copy

    def calculate(self) -> TensorBackend:
        """
        Abstract method for calculation logic.

        Returns:
        - Calculated Tensor object
        """
        raise NotImplementedError("Calculate method not implemented")

    @property
    def code(self) -> str:
        """
        Identifies node for duplicate handling.

        Returns:
        - Code string
        """
        return f"Node at {hex(id(self))}"

    def __repr__(self):
        return self.code
