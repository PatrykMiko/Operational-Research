from copy import deepcopy
from dataclasses import dataclass, field
from saport.linear_programming.exceptions import (
    DuplicateVariableError,
    EmptyModelError,
    MissingObjectiveError,
)
from saport.linear_programming.expressions import (
    Constraint,
    Expression,
    ObjectiveType,
    Objective,
    Variable,
)


@dataclass
class Model:
    """
    A class to represent a linear programming problem.
    """

    name: str
    """name of the model"""

    variables: list[Variable] = field(default_factory=list[Variable])
    """
    list with the model variables, 
    variable with index 'i' is always stored at the variables[i]
    """

    constraints: list[Constraint] = field(default_factory=list[Constraint])
    """list containing model constraints"""

    objective: Objective = field(default_factory=Objective)
    """represents the objective function"""

    def create_variable(self, name: str) -> Variable:
        """
        Returns a new variable with a specified name.
        The variable is automatically indexed and added to the model.
        Variable name has to be unique, otherwise a DuplicateVariableError is raised.

        :param name: name of the variable
        :type name: str
        :return: a new variable
        :rtype: Variable
        """
        for var in self.variables:
            if var.name == name:
                raise DuplicateVariableError(name)

        new_index = len(self.variables)
        variable = Variable(name, new_index)
        self.variables.append(variable)
        return variable

    def add_constraint(self, constraint: Constraint):
        """
        Adds a new constraint to the model.
        The constraint is automatically indexed.

        :param constraint: constraint to be added
        :type constraint: Constraint
        """
        constraint.index = len(self.constraints)
        self.constraints.append(constraint)

    def maximize(self, expression: Expression):
        """
        Overrides the model objective to maximize the specified expression.

        :param expression: an expression to be maximized
        :type expression: Expression
        """
        self.objective = Objective(expression, ObjectiveType.MAX)

    def minimize(self, expression: Expression):
        """
        Overrides the model objective to minimize the specified expression.

        :param expression: an expression to be minimized
        :type expression: Expression
        """
        self.objective = Objective(expression, ObjectiveType.MIN)

    def satisfy(self):
        """
        Overrides the model objective to just satisfy the problem constraints.
        """
        self.objective = Objective(type=ObjectiveType.SAT, coefficient=0.0)

    def simplify(self) -> None:
        """
        Simplifies all expressions in the model
        """
        for c in self.constraints:
            c.simplify()
        self.objective.simplify()

    def validate(self) -> None:
        """
        Checks if the model is correct, i.e.,
        is not empty and has an objective set
        """
        if len(self.variables) == 0:
            raise EmptyModelError()

        if not self.objective.expression.atoms:
            raise MissingObjectiveError()

    def expression_coefficients(self, expression: Expression) -> list[float]:
        """
        Translates an expression from the model to a correctly ordered
        list of the variables occurring in the expression.

        :param expression: expression to be translated
        :type expression: Expression
        :return: list representing the expression's coefficients
        :rtype: list[float]
        """
        simplified_expression = deepcopy(expression)
        simplified_expression.simplify()
        coefficients = [0.0 for _ in self.variables]
        for atom in simplified_expression.atoms:
            if atom.var.index < len(coefficients):
                coefficients[atom.var.index] = atom.coefficient
        return coefficients

    def __str__(self) -> str:
        separator = "\n\t"
        text = (
            f"- name: {self.name},\n"
            f"- variables: {separator}{separator.join([f'{v.name} >= 0' for v in self.variables])},\n"
            f'- constraints: {separator}{separator.join([str(c) for c in self.constraints])}",\n'
            f"- objective: {separator}{self.objective}\n"
        )
        return text
