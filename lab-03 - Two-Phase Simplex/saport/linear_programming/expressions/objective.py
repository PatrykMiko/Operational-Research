from dataclasses import dataclass, field
from typing import Sequence
import saport.linear_programming.expressions.expression as ExpressionM
from saport.linear_programming.expressions.objective_type import ObjectiveType


@dataclass
class Objective:
    """
    A class to represent an objective in the linear programming expression, e.g. 4x + 5y -> max, etc.
    """

    expression: ExpressionM.Expression = field(default_factory=ExpressionM.Expression)
    """polynomial expressions that is being optimized"""
    type: ObjectiveType = ObjectiveType.MIN
    """type of the objective: MIN, MAX, SAT"""
    coefficient: float = 1.0
    """coefficient associated with the objective variable used in simplex algorithm"""

    def invert(self):
        """
        'Inverts' the objective (switches between 'max' and 'min')
        """
        self.expression = self.expression * -1
        self.type = ObjectiveType(self.type.value * -1)
        self.coefficient = self.coefficient * -1

    def simplify(self):
        """
        Simplifies the objective expression in-place.
        """
        self.expression.simplify()

    def evaluate(self, assignment: Sequence[float]) -> float:
        """
        Evaluates the objective given an assignment of values to the involved variables.

        :param assignment:  a list of floats corresponding (by index) to the variables in the model
        :type assignment: Sequence[float]
        :return: returns value of the objective for the given assignment
        :rtype: float
        """
        return self.expression.evaluate(assignment)

    def name(self) -> str:
        """
        Returns a name of the objective variable, including its sign.
        """
        return f"{'-' if self.coefficient < 0 else ''}z"

    def __str__(self) -> str:
        if self.type == ObjectiveType.SAT:
            return "sat"
        return f"{self.type}: {self.name()} = {self.expression}"
