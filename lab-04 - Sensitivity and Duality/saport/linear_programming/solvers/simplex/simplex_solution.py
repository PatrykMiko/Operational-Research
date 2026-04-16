from dataclasses import dataclass

import numpy as np

from saport.linear_programming.solvers.simplex.constants import eps
from saport.linear_programming.expressions.objective_type import ObjectiveType
from saport.linear_programming.expressions.variable import Variable
from saport.linear_programming.solution import Solution
from saport.linear_programming.solvers.simplex.cost_change import CostChange
from saport.linear_programming.solvers.simplex.tableau import Tableau


@dataclass
class SimplexSolution(Solution):
    """
    A more detailed variant of the basic solution class.
    Contains information about the tableau used by a simplex solver.
    """

    initial_tableau: Tableau
    """
    Contains an initial tableau for the solved model.
    Useful for debugging.
    """
    tableau: Tableau
    """
    Contains the final tableau for the solved model.
    """

    def is_unique(self) -> bool:
        """
        Checks whether the optimal solution represented by this solution is unique.

        :return: `True` if the optimum is unique, `False` otherwise
        :rtype: bool
        """
        # TODO: return `True` if and only if all reduced costs of non-basic variables are non-zero
        #
        # 1) check if the tableau is optimal, if not raise a ValueError, since uniqueness can only be checked for an optimal tableau
        # 2) extract the indices of non-basic variables using `extract_non_basis()`
        # 3) extract the reduced costs of non-basic variables from the objective coefficients using the indices of non-basic variables
        # 4) check if any of the reduced costs of non-basic variables is close to zero
        #
        # tip 1. you can use `np.isclose()` to check if the reduced costs are close to zero, with an absolute tolerance of `eps`
        # tip 2. you can use `np.any()` to check if any of the values in an array is True

        if not self.tableau.is_optimal():
            raise ValueError("Cannot check uniqueness for a non-optimal tableau.")

        non_basis = self.tableau.extract_non_basis()
        obj_coeffs = self.tableau.objective_coefficients()
        non_basic_costs = obj_coeffs[non_basis]
        has_zero_cost = np.any(np.isclose(non_basic_costs, 0.0, atol=eps))
        return not has_zero_cost
        #raise NotImplementedError()

    def compute_cost_changes(self) -> list[CostChange]:
        """
        Computes cost changes for the original model variables when no variable preprocessing was needed.

        :return: cost change effects for original model variables
        :rtype: list[CostChange]
        """
        # TODO: compute cost changes for original model variables when no variable preprocessing was needed
        #       the function will use the tableau's method to not compute the cost changes from scratch
        #
        # 1) check if cost changes can be computed using `_validate_cost_changes_supported()`
        # 2) prepare tableau cost changes using `tableau.compute_cost_changes()`
        # 3) extract original variables and their corresponding cost changes from the tableau cost changes
        # 4) convert tableau cost changes to cost changes for the original objective using `_cost_change_for_original_objective()`
        #
        # tip 1. you can use the fact that the original variables are always in the same order as the tableau cost changes
        #        and that they are the first variables in the tableau to extract original variables

        self._validate_cost_changes_supported()
        tableau_cost_changes = self.tableau.compute_cost_changes()
        original_cost_changes = []
        for variable, tableau_cost_change in zip(self.model.variables, tableau_cost_changes):
            original_cost_change = self._cost_change_for_original_objective(
                variable,
                tableau_cost_change
            )
            original_cost_changes.append(original_cost_change)

        return original_cost_changes

        #raise NotImplementedError()

    def _validate_cost_changes_supported(self) -> None:
        unsupported_variables = [
            variable.name
            for variable in self.model.variables
            if not variable.domain.is_normalized
        ]
        if unsupported_variables:
            raise ValueError(
                "SimplexSolution.compute_cost_changes() is only supported when no "
                "variable preprocessing is required. Unsupported variables: "
                + ", ".join(unsupported_variables)
            )

    def _cost_change_for_original_objective(
        self, variable: Variable, cost_change: CostChange
    ) -> CostChange:
        """
        Converts a cost change for a tableau variable to a cost change for the original objective.

        :param variable: original variable for which the cost change is computed
        :type variable: Variable
        :param cost_change: cost change for the corresponding tableau variable
        :type cost_change: CostChange
        :return: a new cost change computed for the original objective
        :rtype: CostChange
        """
        # TODO: convert a cost change for a tableau variable to a cost change for the original objective
        #
        # 1) if the original objective is a minimization, don't negate the cost change's objective delta coefficient and map
        #    (delta_min, delta_max) to (-delta_max, -delta_min)
        # 2) if the original objective is a maximization, return the cost change as is, just change the variable to the original variable
        #
        # tip 1. remember about replacing the variable in the cost change with the one from the original model
        # tip 2. why shouldn't we negate the cost change's objective delta coefficient for a minimization objective?
        #        originally the problem was min z = R + (c + delta)x
        #        which is transformed to max w = -z = -R - (c + delta)x
        #        in tableau we have the coefficient change relative to the transformed objective, so
        #        max w = -R + (-c + delta')x
        #        comparing coefficients gives us delta' = -delta
        #        if the tableau says that the optimal value changes as w = w_opt + b * delta',
        #        then in the original problem
        #        z = -w = -(w_opt + b * delta') = z_opt - b * delta' = z_opt + b * delta
        #        so the objective delta coefficient should NOT be negated here

        is_minimization = self.model.objective.type == ObjectiveType.MIN

        if is_minimization:
            return CostChange(
                variable=variable,
                objective_delta_coefficient=cost_change.objective_delta_coefficient,
                delta_min=-cost_change.delta_max,
                delta_max=-cost_change.delta_min
            )
        else:
            return CostChange(
                variable=variable,
                objective_delta_coefficient=cost_change.objective_delta_coefficient,
                delta_min=cost_change.delta_min,
                delta_max=cost_change.delta_max
            )

        #raise NotImplementedError()
