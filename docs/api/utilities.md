# Utilities API

This section documents the utility components of GIRAFFE.

## Fitness Functions

Functions for evaluating the fitness of trees.

```python
from giraffe.fitness import average_precision_fitness, roc_auc_score_fitness
```

::: giraffe.fitness
    options:
      show_source: true

## Callbacks

The callback system allows customizing the evolutionary process.

```python
from giraffe.callback import Callback
```

::: giraffe.callback.Callback
    options:
      show_source: true

## Visualization

Functions for visualizing trees.

```python
from giraffe.draw import draw_tree
```

::: giraffe.draw.draw_tree
    options:
      show_source: true

## Postprocessing Functions

Functions for postprocessing tree outputs.

```python
from giraffe.functions import scale_vector_to_sum_1, set_multiclass_postprocessing
```

::: giraffe.functions
    options:
      show_source: true

## Other Utilities

Additional utility functions.

```python
from giraffe.utils import Pickle, first_uniques_mask
```

::: giraffe.utils
    options:
      show_source: true
