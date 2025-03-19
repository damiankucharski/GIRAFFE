from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from giraffe.giraffe import Giraffe


class Callback:
    def __init__(self) -> None:
        pass

    def on_generation_end(self, giraffe: Giraffe) -> None:
        pass

    def on_evolution_end(self, giraffe: Giraffe) -> None:
        pass

    def on_evolution_start(self, giraffe: Giraffe) -> None:
        pass

    def on_generation_start(self, giraffe: Giraffe) -> None:
        pass
