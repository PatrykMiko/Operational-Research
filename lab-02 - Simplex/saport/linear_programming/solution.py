from dataclasses import dataclass

from saport.linear_programming.model import Model
from saport.linear_programming.expressions import Variable


@dataclass
class Solution:
    """
    A class to represent a solution to linear programming problem.
    """

    model: Model
    """model which led to the solution"""
    assignment: list[float]
    """a list of values assigned to the model variables"""

    def assignment_for(self, model: Model) -> list[float]:
        """
        Returns an assignment of values to the variables in the specified model.

        :param model: a model containing variables
        :type model: Model
        :return: a list of values assigned to variables in the model
        :rtype: list[float]
        """
        return self.assignment[: len(model.variables)]

    def value(self, var: Variable) -> float:
        """
        Returns a value of the specified variable.

        :param var: a variable
        :type var: Variable
        :return: a value assigned to the given variable
        :rtype: float
        """
        return self.assignment[var.index]

    def objective_value(self) -> float:
        """
        Returns the objective achieved given the solution.

        :return: an objective value
        :rtype: float
        """
        return self.model.objective.evaluate(self.assignment)

    def __str__(self, model: Model | None = None) -> str:
        model = self.model if model is None else model

        text = f"- objective value: {self.objective_value()}\n"
        text += "- assignment:"
        for var in model.variables:
            text += f"\n\t- {var.name} = {'{:.3f}'.format(self.assignment[var.index])}"
        return text
