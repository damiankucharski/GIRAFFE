import pytest
from giraffe.node import Node


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

    assert example_tree["A"].children == [example_tree["C"], replacement]
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
