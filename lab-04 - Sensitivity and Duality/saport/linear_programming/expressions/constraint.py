from dataclasses import dataclass
from saport.linear_programming.expressions.constraint_type import ConstraintType

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from saport.linear_programming.expressions.affine_substitution import (
        AffineSubstitution,
    )
    from saport.linear_programming.expressions.expression import Expression
    from saport.linear_programming.expressions.variable import Variable


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

    def substitute(
        self, substitutions: dict[Variable, AffineSubstitution]
    ) -> Constraint:
        """
        Returns a new constraint with variables replaced by affine substitutions.

        :param substitutions: substitutions to apply
        :type substitutions: dict[Variable, AffineSubstitution]
        :return: a new substituted constraint
        :rtype: Constraint
        """
        expression, constant_shift = self.expression.substitute(substitutions)
        return Constraint(
            expression, self.bound - constant_shift, self.type, self.index
        )

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
