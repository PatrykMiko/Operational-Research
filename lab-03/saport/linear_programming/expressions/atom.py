from dataclasses import dataclass
import saport.linear_programming.expressions.expression as ExpressionM
from typing import Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from saport.linear_programming.expressions.variable import Variable


@dataclass
class Atom(ExpressionM.Expression):
    """
    A class to represent an atom of the linear programming expression, i.e. variable and it's coefficient (e.g. 4x, -5.3x,
    etc.) It derives from the Expression class and can be interpreted as an expression containing only a single atom,
    itself.
    """

    var: Variable
    """variable associated with the atom"""
    coefficient: float
    """coefficient value associated with the atom"""

    def __post_init__(self):
        super().__init__(self)

    def evaluate_with_value(self, assigned_value: float) -> float:
        """
        Returns value of the atom for the given assignment to its variable.

        :param assigned_value: a value assigned to the variable
        :type assigned_value: float
        :return: value of the whole atom
        :rtype: float
        """
        return self.coefficient * assigned_value

    def evaluate(self, assignment: Sequence[float]) -> float:
        return super().evaluate(assignment)

    def __mul__(self, factor: float) -> Atom:
        """
        Returns a new atom with a multiplied coefficient

        :param factor: what should be the atom multiplied by
        :type factor: float
        :return: a new atom with an updated coefficient
        :rtype: Atom
        """
        return Atom(self.var, self.coefficient * factor)

    __rmul__ = __mul__

    def __str__(self):
        if float(self.coefficient) == 1.0:
            return str(self.var)
        elif float(self.coefficient) == -1.0:
            return f"-{self.var}"
        else:
            return f"{self.coefficient}*{self.var}"
