import pytest
from giraffe.node import Node, ValueNode, OperatorNode
import numpy as np


@pytest.fixture
def node_set():
    nset = {
        "A": Node(),
        "B": Node(),
        "C": Node(),
        "D": Node(),
        "E": Node(),
    }
    return nset


@pytest.fixture
def example_tree(node_set):
    """Create a tree with the following structure:
    A
    ├── B
    │   ├── D
    │   └── E
    └── C
    """
    node_set["A"].add_child(node_set["B"])
    node_set["A"].add_child(node_set["C"])
    node_set["B"].add_child(node_set["D"])
    node_set["B"].add_child(node_set["E"])
    return node_set


def test_add_child(example_tree):
    assert example_tree["A"].children == [example_tree["B"], example_tree["C"]]
    assert example_tree["B"].children == [example_tree["D"], example_tree["E"]]
    assert example_tree["C"].children == []
    assert example_tree["D"].children == []
    assert example_tree["E"].children == []

    assert example_tree["A"].parent is None
    assert example_tree["B"].parent is example_tree["A"]
    assert example_tree["C"].parent is example_tree["A"]
    assert example_tree["D"].parent is example_tree["B"]
    assert example_tree["E"].parent is example_tree["B"]


def test_remove_child(example_tree):
    example_tree["A"].remove_child(example_tree["B"])
    example_tree["B"].remove_child(example_tree["D"])

    assert example_tree["A"].children == [example_tree["C"]]
    assert example_tree["B"].children == [example_tree["E"]]
    assert example_tree["C"].children == []
    assert example_tree["D"].children == []
    assert example_tree["E"].children == []

    assert example_tree["A"].parent is None
    assert example_tree["B"].parent is None
    assert example_tree["C"].parent is example_tree["A"]
    assert example_tree["D"].parent is None
    assert example_tree["E"].parent is example_tree["B"]


def test_replace_child_correct(example_tree):
    replacement = Node()

    example_tree["A"].replace_child(example_tree["B"], replacement)

    assert example_tree["A"].children == [replacement, example_tree["C"]]
    assert example_tree["B"].children == [example_tree["D"], example_tree["E"]]
    assert example_tree["C"].children == []
    assert example_tree["D"].children == []
    assert example_tree["E"].children == []
    assert replacement.children == []

    assert example_tree["A"].parent is None
    assert example_tree["B"].parent is None
    assert example_tree["C"].parent is example_tree["A"]
    assert example_tree["D"].parent is example_tree["B"]
    assert example_tree["E"].parent is example_tree["B"]
    assert replacement.parent is example_tree["A"]


def test_replace_child_incorrect(example_tree):
    with pytest.raises(ValueError):
        example_tree["A"].replace_child(example_tree["B"], example_tree["C"])


def test_get_nodes(example_tree):
    nodes = example_tree["A"].get_nodes()
    assert nodes == [example_tree["A"], example_tree["B"], example_tree["C"], example_tree["D"], example_tree["E"]]


def test_copy(example_tree):
    copy = example_tree["A"].copy()
    assert copy.children == []
    assert copy.parent is None


def test_copy_subtree(example_tree):
    copy = example_tree["A"].copy_subtree()
    nodes = copy.get_nodes()

    A, B, C, D, E = nodes

    assert A.children == [B, C]
    assert B.children == [D, E]
    assert C.children == []
    assert D.children == []
    assert E.children == []

    assert A.parent is None
    assert B.parent is A
    assert C.parent is A
    assert D.parent is B
    assert E.parent is B


@pytest.fixture
def value_op_base_set():

    def x(): return np.array([[2,2],[3,3]])

    nset = {
        "A": ValueNode(None, None, x(), 1),
        "B": OperatorNode(None, None),
        "C": ValueNode(None, None, x(), 2),
        "D": ValueNode(None, None, x(), 3),
    }

    return nset

@pytest.fixture
def value_op_base_tree(value_op_base_set):
    """Create a tree with the following structure:
    A
    ├── B
    │   ├── D
    │   └── C
    """
    value_op_base_set["B"].add_child(value_op_base_set["D"])
    value_op_base_set["B"].add_child(value_op_base_set["C"])
    value_op_base_set["A"].add_child(value_op_base_set["B"])
    return value_op_base_set

def test_concat(value_op_base_tree):
    concat = value_op_base_tree["B"]._concat()
    assert np.array_equal(concat.shape(), (3,2,2))