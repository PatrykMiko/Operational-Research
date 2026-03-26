from copy import deepcopy
import numpy as np

from saport.linear_programming.result import Result
from saport.linear_programming.expressions import (
    Variable,
    ObjectiveType,
    ConstraintType,
    Constraint,
)
from saport.linear_programming.model import Model
from saport.linear_programming.solver import Solver as LinearProgrammingSolver
from saport.linear_programming.solvers.simplex.constants import eps
from saport.linear_programming.solvers.simplex.simplex_solution import SimplexSolution
from saport.linear_programming.solvers.simplex.tableau import Tableau


class SimplexSolver(LinearProgrammingSolver[SimplexSolution]):
    """
    A class to represent a simplex solver.
    """

    _slacks: dict[Variable, Constraint]
    """contains mapping from slack variables to their corresponding constraints"""
    _surpluses: dict[Variable, Constraint]
    """contains mapping from surplus variables to their corresponding constraints"""
    _artificial: dict[Variable, Constraint]
    """contains mapping from artificial variables to their corresponding constraints"""

    def solve(self) -> Result[SimplexSolution]:
        """
        Solves the given linear model.

        :param model: a linear programming model to be solved
        :type model: Model
        :return: an object representing the solution
        :rtype: Solution
        """
        self.model.validate()
        normal_model = self._augment_model(self.model)

        tableau = self._basic_initial_tableau(normal_model)

        initial_tableau = deepcopy(tableau)
        if not self._optimize(tableau):
            return Result[SimplexSolution].unbounded()

        assignment = tableau.extract_assignment()
        model_assignment = [assignment[var.index] for var in self.model.variables]
        return Result[SimplexSolution].with_solution(
            SimplexSolution(self.model, model_assignment, initial_tableau, tableau)
        )

    def _optimize(self, tableau: Tableau) -> bool:
        """
        An internal method executing the simplex loop.

        :param tableau: a starting point (tableau) for the simplex method
        :type tableau: Tableau
        :return: `True` if a solution has been found, `False` if there is no optimal solution
        :rtype: bool
        """
        while not tableau.is_optimal():
            pivot_col = tableau.choose_entering_variable()
            if tableau.is_unbounded(pivot_col):
                return False
            pivot_row = tableau.choose_leaving_variable(pivot_col)

            tableau.pivot(pivot_row, pivot_col)
        return True

    def _augment_model(self, original_model: Model) -> Model:
        """
        Creates an augmented model, satisfying the simplex method requirements.

        :param original_model: a linear model as defined by the user
        :type original_model: Model
        :return: a simplex-friendly model (max objective, <= constraints, nonnegative bounds)
        :rtype: Model
        """
        model = deepcopy(original_model)
        model.simplify()
        self._change_objective_to_max(model)
        self._change_constraints_bounds_to_nonnegative(model)
        self._add_slack_and_surplus_variables(model)
        return model

    def _change_objective_to_max(self, model: Model) -> None:
        """
        Modifies a given model in-place to have a "maximize" objective

        :param model: the model to be modified
        :type model: Model
        """
        if model.objective.type == ObjectiveType.MIN:
            model.objective.invert()

    def _change_constraints_bounds_to_nonnegative(self, model: Model):
        """
        Modifies a given model in-place to have only non-negative bounds.

        :param model: a model to be modified
        :type model: Model
        """
        for constraint in model.constraints:
            if constraint.bound < 0:
                constraint.invert()

    def _add_slack_and_surplus_variables(self, model: Model) -> None:
        """
        Augments a given model with slack and surplus variables.
        The new variables are then stored in the self._slacks and self._surpluses dictionaries,
        mapping them to their corresponding constraints.

        :param model: a model to be augmented
        :type model: Model
        """
        slacks: dict[Variable, Constraint] = dict()
        surpluses: dict[Variable, Constraint] = dict()

        for constraint in model.constraints:
            if constraint.type == ConstraintType.LE:
                slack_var = model.create_variable(f"s{constraint.index}")
                slacks[slack_var] = constraint
                constraint.expression += slack_var
                constraint.type = ConstraintType.EQ
            if constraint.type == ConstraintType.GE:
                surplus_var = model.create_variable(f"s{constraint.index}")
                surpluses[surplus_var] = constraint
                constraint.expression -= surplus_var
                constraint.type = ConstraintType.EQ

        self._slacks = slacks
        self._surpluses = surpluses

    def _basic_initial_tableau(self, model: Model) -> Tableau:
        """
        Creates a trivial initial tableau for the models satisfying some basic requirements.

        :param model: a model having a trivial initial solution
        :type model: Model
        :return: an trivial initial tableau
        :rtype: Tableau
        """
        objective_row = -1 * np.array(
            model.objective.expression.get_coefficients(model.variables) + [0.0]
        )
        table = np.array(
            [objective_row]
            + [
                c.expression.get_coefficients(model.variables) + [c.bound]
                for c in model.constraints
            ]
        )
        return Tableau(model, table)
