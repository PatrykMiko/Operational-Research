from dataclasses import dataclass
from saport.linear_programming.expressions.atom import Atom


@dataclass(init=False)
class Variable(Atom):
    """
    A class to represent a linear programming variable.
    It derives from the Atom class and can be interpreted as an Atom with factor = 1.
    """

    name: str
    """name of the variable"""

    index: int
    "index of the variable used in the model"

    def __init__(self, name: str, index: int):
        """
        Initializes the variable.

        :param name: a name of the variable
        :type name: str
        :param index: the variable's index, unique given the model
        :type index: int
        """
        self.name = name
        self.index = index
        super().__init__(self, 1)

    def __str__(self) -> str:
        return self.name

    def __key__(self) -> tuple[str, int]:
        return self.name, self.index

    def __hash__(self) -> int:
        return hash(self.__key__())

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        if not isinstance(other, Variable):
            return NotImplemented

        return self.__key__() == other.__key__()
