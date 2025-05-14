# Nodes API

Nodes are the fundamental building blocks of trees in GIRAFFE. This section documents the different types of nodes available.

## Base Node Class

The `Node` class is the base class for all node types in GIRAFFE.

```python
from giraffe.node import Node
```

::: giraffe.node.Node
    options:
      show_root_heading: false
      show_source: true

## Value Node

The `ValueNode` class represents a node that holds tensor data (model predictions).

```python
from giraffe.node import ValueNode
```

::: giraffe.node.ValueNode
    options:
      show_root_heading: false
      show_source: true

## Operator Node

The `OperatorNode` class is the base class for all operation nodes that define how to combine tensor data.

```python
from giraffe.node import OperatorNode
```

::: giraffe.node.OperatorNode
    options:
      show_root_heading: false
      show_source: true

## Specific Operator Nodes

Various specific operator node implementations are provided:

### Mean Node

::: giraffe.node.MeanNode
    options:
      show_root_heading: false
      show_source: true

### Weighted Mean Node

::: giraffe.node.WeightedMeanNode
    options:
      show_root_heading: false
      show_source: true

### Min Node

::: giraffe.node.MinNode
    options:
      show_root_heading: false
      show_source: true

### Max Node

::: giraffe.node.MaxNode
    options:
      show_root_heading: false
      show_source: true

## Utility Functions

::: giraffe.node.check_if_both_types_values
    options:
      show_source: true

::: giraffe.node.check_if_both_types_operators
    options:
      show_source: true

::: giraffe.node.check_if_both_types_same_node_variant
    options:
      show_source: true
