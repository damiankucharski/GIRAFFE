import pytest
from giraffe.node import ValueNode, OperatorNode
from giraffe.tree import Tree
import numpy as np


@pytest.fixture
def value_op_base_set():
    def x():
        return np.array([[2, 2], [3, 3]])
    
    nset = {
            "A": ValueNode(None, x(), "A"),
            "B": OperatorNode(None),
            "C": OperatorNode(None),
            "D": ValueNode(None, x(), "D"),
            "E": ValueNode(None, x(), "E"),
            "F": ValueNode(None, x(), "F"),
            "G": ValueNode(None, x(), "G"),
            "H": ValueNode(None, x(), "H"),
        }

    return nset


@pytest.fixture
def two_base_trees(value_op_base_set):
    r"""
    Creates two trees with the following structure:
    Tree 1:
         A
         |
         B
        /|\
       / | \
      D  E  G

    Tree 2:
         F
         |
         C
         |
         H
    """

    value_op_base_set["A"].add_child(value_op_base_set["B"])
    value_op_base_set["B"].add_child(value_op_base_set["D"])
    value_op_base_set["B"].add_child(value_op_base_set["E"])
    value_op_base_set["B"].add_child(value_op_base_set["G"])


    value_op_base_set["F"].add_child(value_op_base_set["C"])
    value_op_base_set["C"].add_child(value_op_base_set["H"])

    tree1 = Tree.create_tree_from_root(value_op_base_set["A"])
    tree2 = Tree.create_tree_from_root(value_op_base_set["F"])

    return tree1, tree2


def test_tree_nodes_lists(two_base_trees):

    tree1, tree2 = two_base_trees

    assert len(tree1.nodes["value_nodes"]) == 4
    assert len(tree1.nodes["op_nodes"]) == 1
    
    assert len(tree2.nodes["value_nodes"]) == 2
    assert len(tree2.nodes["op_nodes"]) == 1