from copy import deepcopy
import numpy as np

from saport.linear_programming.expressions.expression import Expression
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
from saport.linear_programming.transformations import preprocess_variables


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
        prepared_model = preprocess_variables(self.model)
        normal_model = self._augment_model(prepared_model.model)
        if len(self._slacks) < len(normal_model.constraints):
            tableau, success = self._presolve(normal_model)
            if not success:
                return Result[SimplexSolution].infeasible()
        else:
            tableau = self._basic_initial_tableau(normal_model)

        initial_tableau = deepcopy(tableau)
        if not self._optimize(tableau):
            return Result[SimplexSolution].unbounded()

        assignment = tableau.extract_assignment()
        model_assignment = prepared_model.restore_assignment(assignment)
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

    def _presolve(self, model: Model) -> tuple[Tableau, bool]:
        """
        Finds an initial solution/tableau for a given problem.

        :param model: a linear model to be solved
        :type model: Model
        :return: If the second element of the tuple is `True`, then a solution has been found
                 otherwise, problem is infeasible. The first element is the corresponding tableau.
        :rtype: tuple[Tableau, bool]
        """
        presolve_model = self._create_presolve_model(model)
        tableau = self._presolve_initial_tableau(presolve_model)

        self._optimize(tableau)

        if self._artificial_variables_are_positive(tableau):
            return (tableau, False)

        tableau = self._restore_initial_tableau(tableau, model)
        return (tableau, True)

    def _augment_model(self, model: Model) -> Model:
        """
        Creates an augmented model, satisfying the simplex method requirements.

        :param model: a normalized model as required by the simplex method
        :type model: Model
        :return: a simplex-friendly model (max objective, <= constraints, nonnegative bounds)
        :rtype: Model
        """
        model.simplify()
        self._change_objective_to_max(model)
        self._change_constraints_bounds_to_nonnegative(model)
        self._add_slack_and_surplus_variables(model)
        return model

    def _create_presolve_model(self, augmented_model: Model) -> Model:
        """
        Creates an artificial model, designed to find an initial solution

        :param augmented_model: a linear model as required by the simplex algorithm
        :type original_model: Model
        :return: a model with a trivial initial solution
        :rtype: Model
        """
        presolve_model = deepcopy(augmented_model)
        self._artificial = self._add_artificial_variables(presolve_model)
        goal_expression = sum([-var for var in self._artificial], Expression())
        assert goal_expression is not None
        presolve_model.maximize(goal_expression)
        return presolve_model

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

    def _add_artificial_variables(self, model: Model) -> dict[Variable, Constraint]:
        """
        Relaxes a given model in-place by introducing artificial variables.

        :param model: a model to be relaxed
        :type model: Model
        :return:  a dictionary mapping new variables to their corresponding constraints
        :rtype: dict[Variable, Constraint]
        """
        artificial_variables: dict[Variable, Constraint] = dict()
        for constraint in model.constraints.copy():
            if (
                len([c for c in self._slacks.values() if c.index == constraint.index])
                > 0
            ):
                continue
            artificial_var = model.create_variable(f"R{constraint.index}")
            artificial_variables[artificial_var] = constraint
            constraint.expression = constraint.expression + artificial_var
        return artificial_variables

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

    def _presolve_initial_tableau(self, model: Model) -> Tableau:
        """
        Creates a trivial initial tableau for the presolve model.

        :param model: a model containing artificial variables
        :type model: Model
        :return: an trivial initial tableau
        :rtype: Tableau
        """
        tableau = self._basic_initial_tableau(model)

        for var, constraint in self._artificial.items():
            tableau.pivot(constraint.index + 1, var.index)

        return tableau

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

    def _artificial_variables_are_positive(self, tableau: Tableau) -> bool:
        """
        Checks whether there is at least one artificial variable with a positive value in the tableau.

        :param tableau: a tableau for an augmented model
        :type tableau: Tableau
        :return: `True` if at least one artificial variable is positive, else `False`
        :rtype: bool
        """

        assignment = tableau.extract_assignment()
        for artificial_var in self._artificial:
            if assignment[artificial_var.index] > eps:
                return True
        return False

    def _restore_initial_tableau(
        self, tableau: Tableau, original_model: Model
    ) -> Tableau:
        """
        Restores a tableau of the presolve model to the original model, removing the artificial variables.

        :param tableau: a tableau of the presolve model, with all artificial variables equal zero
        :type tableau: Tableau
        :param model: the original linear model, a.k.a., augmented model
        :type model: Model
        :return: a new tableau for the original model
        :rtype: Tableau
        """
        basis = tableau.extract_basis()
        tableau = self._remove_artificial_variables(tableau)
        tableau = self._restore_original_objective_row(tableau, original_model)
        tableau = self._fix_objective_row_to_the_basis(tableau, basis)
        return tableau

    def _remove_artificial_variables(self, tableau: Tableau) -> Tableau:
        """
        Removes artificial variables from the tableau.
        The operation is not done in-place, a new tableau is returned.

        :param tableau: tableau with artificial variables
        :type tableau: Tableau
        :return: a new tableau without the artificial variables
        :rtype: Tableau
        """
        columns_to_remove = [var.index for var in self._artificial.keys()]
        table = np.delete(tableau.table, columns_to_remove, 1)
        return Tableau(tableau.model, table)

    def _restore_original_objective_row(
        self, tableau: Tableau, original_model: Model
    ) -> Tableau:
        """
        Restores an objective row in the presolve model tableau to the original model.

        :param tableau: tableau with a "presolve" objective
        :type tableau: Tableau
        :param original_model: an original "augmented" model
        :type original_model: Model
        :return: a new tableau with an original objective
        :rtype: Tableau
        """
        basic_tableau = self._basic_initial_tableau(original_model)
        new_table = np.array(tableau.table)
        new_table[0] = basic_tableau.table[0]
        return Tableau(original_model, new_table)

    def _fix_objective_row_to_the_basis(
        self, tableau: Tableau, basis: list[int]
    ) -> Tableau:
        """
        Fixes the objective row, so the tableau has a proper basis

        :param tableau: tableau with a "presolve" objective
        :type tableau: Tableau
        :param basis: indices of the variables that should be in basis
        :param list[int]
        :return: a new tableau with a proper basis
        :rtype: Tableau
        """
        new_tableau = deepcopy(tableau)
        for constr_index, var_index in enumerate(basis):
            new_tableau.pivot(constr_index + 1, var_index)
        return new_tableau
