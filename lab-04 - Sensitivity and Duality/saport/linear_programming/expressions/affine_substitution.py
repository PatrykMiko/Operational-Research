from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from saport.linear_programming.expressions.expression import Expression


@dataclass(frozen=True)
class AffineSubstitution:
    """
    A class representing an affine substitution of a variable with an expression and an offset.
    """

    expression: Expression
    """expression used in the substitution"""

    offset: float = 0.0
    """constant term added to the expression"""

    def evaluate(self, assignment: Sequence[float]) -> float:
        """
        Evaluates the substitution for a given assignment.

        :param assignment: values aligned with the substituted model variables
        :type assignment: Sequence[float]
        :return: reconstructed value of the original variable
        :rtype: float
        """
        return self.offset + self.expression.evaluate(assignment)
