# Evolution API

This section documents the evolutionary components of GIRAFFE.

## Population

Functions for initializing and managing populations of trees.

```python
from giraffe.population import initialize_individuals, choose_n_best, choose_pareto, choose_pareto_then_sorted
```

::: giraffe.population
    options:
      show_source: true

## Crossover

Functions for performing crossover between trees.

```python
from giraffe.crossover import crossover, tournament_selection_indexes
```

::: giraffe.crossover
    options:
      show_source: true

## Mutation

Functions for mutating trees.

```python
from giraffe.mutation import append_new_node_mutation, lose_branch_mutation, new_tree_from_branch_mutation, get_allowed_mutations
```

::: giraffe.mutation
    options:
      show_source: true

## Pareto Optimization

Functions for Pareto optimization and visualization.

```python
from giraffe.pareto import paretoset, minimize, maximize, plot_pareto_frontier
```

::: giraffe.pareto
    options:
      show_source: true
