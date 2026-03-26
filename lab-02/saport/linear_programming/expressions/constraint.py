from dataclasses import dataclass
from saport.linear_programming.expressions.constraint_type import ConstraintType

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from saport.linear_programming.expressions.expression import Expression


@dataclass
class Constraint:
    """
    A class to represent a constraint in the linear programming expression, e.g. 4x + 5y <= 13, etc.
    """

    expression: Expression
    """
    the left side of the equation — a linear polynomial
    """

    bound: float
    """
    the right side of the equation — a 'real-number' bound 
    """

    type: ConstraintType = ConstraintType.GE
    """
    type of the equation/inequality representing the constraint
    """

    index: int = -1
    """
    index of the constraint in the model, equals `-1` when not added to a model
    """

    def simplify(self):
        """
        Simplifies the left side of the equation in-place.
        """
        self.expression.simplify()

    def invert(self):
        """
        Multiplies the equation times `-1`.
        """
        self.type = ConstraintType(self.type.value * -1)
        self.expression = self.expression * -1
        self.bound = self.bound * -1

    def __str__(self):
        return f"{self.expression} {self.type} {self.bound}"
