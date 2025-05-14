from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from giraffe.giraffe import Giraffe


class Callback:
    """
    Base class for callbacks that can be triggered during the evolutionary process.

    Callbacks allow monitoring and potentially modifying the evolution process at
    specific points: before/after each generation, and at the start/end of the
    entire evolution. Custom callbacks should inherit from this class and override
    the methods corresponding to the desired intervention points.
    """

    def __init__(self) -> None:
        pass

    def on_generation_end(self, giraffe: "Giraffe") -> None:
        """
        Called at the end of each generation.

        This hook is triggered after a generation (iteration) of evolution has completed,
        including selection, crossover, and mutation operations.

        Args:
            giraffe: The Giraffe instance running the evolution
        """
        pass

    def on_evolution_end(self, giraffe: "Giraffe") -> None:
        """
        Called at the end of the entire evolution process.

        This hook is triggered when all generations have been completed or
        when the evolution process is manually stopped.

        Args:
            giraffe: The Giraffe instance running the evolution
        """
        pass

    def on_evolution_start(self, giraffe: "Giraffe") -> None:
        """
        Called at the start of the evolution process.

        This hook is triggered before any generations are run, after the
        initial population has been created.

        Args:
            giraffe: The Giraffe instance running the evolution
        """
        pass

    def on_generation_start(self, giraffe: "Giraffe") -> None:
        """
        Called at the start of each generation.

        This hook is triggered before a generation (iteration) of evolution begins,
        before any selection, crossover, or mutation operations.

        Args:
            giraffe: The Giraffe instance running the evolution
        """
        pass


class FitnessNoChangeEarlyStoppingCallback(Callback):
    def __init__(self, n_iterations=5):
        super().__init__()
        self._iterations_no_change = 0
        self._last_fitnesses = None
        self._n_iterations = n_iterations

    def on_generation_end(self, giraffe: "Giraffe") -> None:
        if self._last_fitnesses is None:
            self._last_fitnesses = giraffe.fitnesses
            return
        assert isinstance(giraffe.fitnesses, np.ndarray), "Fitnesses need to be numpy array"
        if np.allclose(self._last_fitnesses, giraffe.fitnesses):
            self._iterations_no_change += 1
            if self._iterations_no_change >= self._n_iterations:
                giraffe.should_stop = True
        else:
            self._iterations_no_change = 0
