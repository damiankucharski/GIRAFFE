from typing import List, Optional, Union, Callable, cast, Sequence

class Node:
    """
    Nodes act as the fundamental building blocks of a tree,
    capable of holding children and a reference to their parent node.
    """

    def __init__(self, parent: Optional["Node"] = None, children: Optional[Sequence["Node"]] = None):
        self.parent = parent
        self.children: List[Node] = list(children) if children is not None else []
        self.type = None

    def add_child(self, child_node: "Node"):
        """
        Add a child to the Node.

        Parameters:
        - child_node: Node to be added as child
        """
        self.children.append(child_node)
        child_node.parent = self

    def remove_child(self, child_node: "Node") -> "Node":
        self.children.remove(child_node)
        child_node.parent = None
        return child_node

    def replace_child(self, child, replacement_node):
        if replacement_node.parent is not None:
            raise ValueError("Replacement node already has a parent")

        ix = self.children.index(child)
        self.children[ix] = replacement_node

        child.parent = None
        replacement_node.parent = self

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
        return Node() 

    def copy_subtree(self):
        """
        Copy the subtree rooted at this node.
        Returns:
        - Copy of the subtree rooted at this node
        """
        self_copy = self.copy()
        
        for child in self.children:
            child_copy = child.copy_subtree()
            self_copy.add_child(child_copy)
        
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


class ValueNode(Node):
    """
    Represents a Value Node in a computational tree.

    A Value Node holds a specific value or tensor.
    """

    def __init__(self, parent: Optional[Node], children: Optional[Sequence[Node]], value: TensorBackend, id: Union[int, str]):
        super().__init__(parent, children)
        self.value = value
        self.evaluation: Union[None, TensorBackend] = None
        self.id = id

    def calculate(self):
        if self.children:
            for child in self.children:
                self.evaluation = child.calculate()
        else:
            self.evaluation = self.value
        return self.evaluation

    def __str__(self):
        return f"ValueNode with value at: {hex(id(self.value))}"  # and evaluation: {self.evaluation}"

    def add_child(self, child_node):
        super().add_child(child_node)
        self.evaluation = None

    def copy(self):
        return ValueNode(None, None, self.value, self.id)

    @property
    def code(self) -> str:
        return f"VN[{self.id}]"
    

class OperatorNode(Node):
    """
    Abstract Base Class for an Operator Node in a computational tree.

    Reduction Operator Nodes are specialized Operator Nodes capable
    of performing reduction operations like mean, max, min, etc., on tensors.
    """

    def __init__(
        self,
        parent: Optional[ValueNode],
        children: Optional[Sequence[ValueNode]],
        operator: Callable[[TensorBackend], TensorBackend] = lambda x: x,
    ):
        
        super().__init__(parent, children)
        self.operator = operator

