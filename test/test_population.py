from typing import List, cast

import numpy as np

from giraffe.node import ValueNode
from giraffe.population import choose_n_best, choose_pareto, choose_pareto_then_sorted
from giraffe.tree import Tree


class MockTree(Tree):
    """A subclass of Tree that allows setting a mock nodes_count"""

    def __init__(self, root, mock_nodes_count):
        super().__init__(root)
        self._mock_nodes_count = mock_nodes_count

    @property
    def nodes_count(self):
        return self._mock_nodes_count


# Helper function to create mock trees for testing
def create_mock_tree(id_value, nodes_count):
    """Create a mock tree with specified ID and node count for testing"""
    # Create a simple root node with a tensor value
    root = ValueNode(children=None, value=np.array([0.5]), id=id_value)

    # Instead of using Tree directly, use our MockTree subclass
    tree = MockTree(root, nodes_count)

    return tree


class TestSelectionFunctions:
    def setup_method(self):
        """Set up test data before each test method execution"""
        # Create sample trees with different IDs, and mock different node counts
        self.trees = [
            create_mock_tree("tree1", 10),  # Good fitness, medium complexity
            create_mock_tree("tree2", 5),  # Good fitness, low complexity
            create_mock_tree("tree3", 15),  # Medium fitness, high complexity
            create_mock_tree("tree4", 8),  # Low fitness, medium complexity
            create_mock_tree("tree5", 3),  # Medium fitness, low complexity
        ]

        # Define fitness values for each tree
        self.fitnesses = np.array([0.9, 0.85, 0.7, 0.5, 0.65])

    def test_choose_n_best(self):
        """Test that choose_n_best selects trees with highest fitness values"""
        # Test selecting 3 trees
        # Cast self.trees to List[Tree] to satisfy type checker
        trees_for_selection = cast(List[Tree], self.trees)
        selected_trees, selected_fitnesses = choose_n_best(trees_for_selection, self.fitnesses, 3)

        # Check correct number of trees returned
        assert len(selected_trees) == 3
        assert len(selected_fitnesses) == 3

        # Verify trees with highest fitness are selected
        assert selected_trees[0].root.id == "tree1"  # Highest fitness
        assert selected_trees[1].root.id == "tree2"  # Second highest
        assert selected_trees[2].root.id == "tree3"  # Third highest

        # Check fitness values are correct
        assert selected_fitnesses[0] == 0.9
        assert selected_fitnesses[1] == 0.85
        assert selected_fitnesses[2] == 0.7

        # Test selecting 1 tree
        selected_trees, selected_fitnesses = choose_n_best(cast(List[Tree], self.trees), self.fitnesses, 1)
        assert len(selected_trees) == 1
        assert selected_trees[0].root.id == "tree1"  # Only highest fitness
        assert selected_fitnesses[0] == 0.9

    def test_choose_pareto(self):
        """Test that choose_pareto selects trees based on Pareto optimality"""
        # Test selecting up to 3 trees based on Pareto front
        selected_trees, selected_fitnesses = choose_pareto(cast(List[Tree], self.trees), self.fitnesses, 3)

        # In our test case, the Pareto-optimal trees should be:
        # - tree2: High fitness (0.85), low nodes (5)
        # - tree1: Highest fitness (0.9), medium nodes (10)
        # - tree5: Medium fitness (0.65), lowest nodes (3)

        # Check correct trees were selected
        selected_ids = [tree.root.id for tree in selected_trees]
        assert "tree1" in selected_ids
        assert "tree2" in selected_ids
        assert "tree5" in selected_ids

        # Verify fitness values match the selected trees
        for tree, fitness in zip(selected_trees, selected_fitnesses, strict=True):
            expected_fitness = self.fitnesses[self.trees.index(tree)]
            assert fitness == expected_fitness

        # Test with smaller selection limit than Pareto front size
        selected_trees, selected_fitnesses = choose_pareto(cast(List[Tree], self.trees), self.fitnesses, 2)

        # Should select the 2 best by fitness from the Pareto front
        assert len(selected_trees) == 2
        selected_ids = [tree.root.id for tree in selected_trees]
        assert "tree1" in selected_ids  # Highest fitness
        assert "tree2" in selected_ids  # Second highest fitness

    def test_choose_pareto_then_sorted(self):
        """Test choose_pareto_then_sorted selects Pareto-optimal trees first, then fills with best remaining"""
        # Test selecting 4 trees (3 Pareto + 1 from remaining)
        selected_trees, selected_fitnesses = choose_pareto_then_sorted(cast(List[Tree], self.trees), self.fitnesses, 4)

        # First check we got the right number of trees
        assert len(selected_trees) == 4
        assert len(selected_fitnesses) == 4

        # Check all Pareto-optimal trees are included
        selected_ids = [tree.root.id for tree in selected_trees]
        assert "tree1" in selected_ids
        assert "tree2" in selected_ids
        assert "tree5" in selected_ids

        # The fourth tree should be the best remaining by fitness (tree3)
        assert "tree3" in selected_ids

        # Test with limit smaller than Pareto front
        selected_trees, selected_fitnesses = choose_pareto_then_sorted(cast(List[Tree], self.trees), self.fitnesses, 2)

        # Should select the 2 best by fitness from the Pareto front
        assert len(selected_trees) == 2
        selected_ids = [tree.root.id for tree in selected_trees]
        assert "tree1" in selected_ids  # Highest fitness
        assert "tree2" in selected_ids  # Second highest fitness

        # Test with limit larger than available trees
        selected_trees, selected_fitnesses = choose_pareto_then_sorted(cast(List[Tree], self.trees), self.fitnesses, 10)

        # Should include all trees
        assert len(selected_trees) == 5
        assert len(selected_fitnesses) == 5
