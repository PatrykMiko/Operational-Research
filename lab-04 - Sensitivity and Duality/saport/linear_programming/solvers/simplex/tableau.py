import math
import numpy as np
from numpy.typing import NDArray
from saport.linear_programming.expressions.variable import Variable
from saport.linear_programming.model import Model
from saport.linear_programming.solvers.simplex.cost_change import CostChange
from saport.linear_programming.solvers.simplex.constants import eps
from dataclasses import dataclass


TableauIndex = np.intp | int
"""this type alias solves typing issues, when mixing python and numpy types"""


@dataclass
class Tableau:
    """
    A class representing a tableau for a simplex method.
    """

    model: Model
    """model corresponding to the tableaux"""
    table: NDArray[np.double]
    """2d-array containing the tableau"""

    def objective_coefficients(self) -> NDArray[np.double]:
        """
        Returns the first row of the tableau, corresponding to the objective.
        It does not include the objective value.

        :return: objective coefficients — a vector of double
        :rtype: NDArray[double]
        """
        return self.table[0, :-1]

    def objective(self) -> np.double:
        """
        Return the current value of the objective (right-top corner of the tableau)

        :return: current objective value
        :rtype: double
        """
        return self.table[0, -1]

    def is_optimal(self) -> bool:
        """
        Checks, whether the current tableau represents an optimal solution.

        :return: `True` if the solution is optimal, `False` otherwise
        :rtype: bool
        """
        return self.objective_coefficients().min() >= -eps

    def choose_entering_variable(self) -> TableauIndex:
        """
        Finds an index of the variable that should enter the basis.

        :return: an index of the entering variable (column)
        :rtype: TableauIndex
        """
        return self.objective_coefficients().argmin()

    def is_unbounded(self, col: TableauIndex) -> bool:
        """
        Checks, whether the current tableau represent an unbounded problem
        given the selected entering variable.

        :param col: index of an entering variable
        :type col: TableauIndex
        :return: `True` if the problem is unbounded, `False` otherwise
        :rtype: bool
        """
        return self.table[1:, col].max() <= 0

    def choose_leaving_variable(self, col: TableauIndex) -> TableauIndex:
        """
        Finds an index of the variable that should leave the basis given
        the entering variable index.

        :param col: index of the entering variable
        :type col: TableauIndex
        :return: an index of the leaving variable (column)
        :rtype: TableauIndex
        """
        coeffs = self.table[1:, col]
        masked_coeffs = np.ma.masked_less_equal(coeffs, eps)
        bound_col = self.table[1:, -1]
        ratios = bound_col / masked_coeffs
        return np.argmin(ratios) + 1

    def pivot(self, row: TableauIndex, col: TableauIndex) -> None:
        """
        Performs a pivot operation given the coordinates of the entering and leaving variables.
        It modifies the tableau in-place and doesn't return anything.

        :param row: the row-coordinate represents the leaving variable
        :type row: TableauIndex
        :param col: the col-coordinate represents the entering variable
        :type col: TableauIndex
        """
        rows_n, cols_n = self.table.shape
        pivot_factor = self.table[row, col]

        self.table[row] /= pivot_factor

        indices_to_change = np.arange(rows_n) != row
        mult_rows = (
            self.table[indices_to_change, col, np.newaxis] * self.table[row, np.newaxis]
        )
        self.table[indices_to_change] -= mult_rows

    def extract_assignment(self) -> list[float]:
        """
        Extract a value assignment represented by the current state of tableau.

        :return: a list of values assigned to the variables in the tableau
        :rtype: list[float]
        """
        rows_n, cols_n = self.table.shape
        assignment = [0.0 for _ in range(cols_n - 1)]
        basis = self.extract_basis()
        for r in range(1, rows_n):
            var_index = basis[r - 1]
            assignment[var_index] = self.table[r, -1]

        return assignment

    def extract_basis(self) -> list[int]:
        """
        Extracts variables belonging to the basis.

        :return: a list of indices of the variables from the basis
        :rtype: list[int]
        """
        rows_n, cols_n = self.table.shape
        basis = [-1 for _ in range(rows_n - 1)]
        for c in range(cols_n - 1):
            column = self.table[:, c]
            belongs_to_basis = (
                math.isclose(column.min(), 0.0, abs_tol=eps)
                and math.isclose(column.max(), 1.0, abs_tol=eps)
                and math.isclose(column.sum(), 1.0, abs_tol=eps)
            )
            if belongs_to_basis:
                row = np.where(column == 1.0)[0][0]
                # [row-1] because we ignore the cost variable in the basis
                basis[row - 1] = c
        return basis

    def extract_non_basis(self) -> list[int]:
        """
        Extracts variables not belonging to the basis.

        :return: a list of indices of the variables outside of the basis
        :rtype: list[int]
        """
        basis = np.array(self.extract_basis(), dtype=np.intp)
        variables_n = len(self.model.variables)

        if np.any(basis < 0):
            raise ValueError("Cannot extract non-basis from an incomplete basis")

        all_indices = np.arange(variables_n, dtype=np.intp)
        is_nonbasic = np.ones(variables_n, dtype=bool)
        is_nonbasic[basis] = False
        return all_indices[is_nonbasic].tolist()


    def compute_cost_changes(self) -> list[CostChange]:
        """
        Computes the allowable cost coefficient changes for tableau variables.

        :return: cost change effects for all tableau variables
        :rtype: list[CostChange]
        """
        # TODO: compute the cost change effects for all variables in the tableau, including both basic and non-basic variables
        #
        # 1) check if the tableau is optimal, if not raise a ValueError, since cost changes can only be computed for an optimal tableau
        # 2) extract the indices of basic and non-basic variables using `extract_basis()` and `extract_non_basis()`
        # 3) create a mapping for basic variables (for example, a dictionary) from variable indices to their corresponding row indices in the tableau 
        # 4) for each variable in the model, check if it is basic or non-basic using the indices of basic and non-basic variables
        #   a) if the variable is basic, compute its cost change effect using the `_compute_cost_change_in_basis()` method
        #   b) if the variable is non-basic, compute its cost change effect using the `_compute_cost_change_out_of_basis()` method
        # 5) collect the cost change effects for all variables in a list and return it

        if not self.is_optimal():
            raise ValueError("Cannot compute cost change effects for a non-optimal tableau.")

        basis = self.extract_basis()
        non_basis = self.extract_non_basis()
        variable_to_constraint_index = {var_idx: row_idx for row_idx, var_idx in enumerate(basis)}
        all_variables_indices = sorted(list(basis) + list(non_basis))
        cost_changes = []
        for var_idx in all_variables_indices:
            variable_obj = self.model.variables[var_idx]
            if var_idx in variable_to_constraint_index:
                change = self._compute_cost_change_in_basis(
                    variable_obj,
                    non_basis,
                    variable_to_constraint_index
                )
            else:
                change = self._compute_cost_change_out_of_basis(variable_obj)
            cost_changes.append(change)
        return cost_changes

        raise NotImplementedError()

    def _compute_cost_change_in_basis(
        self,
        variable: Variable,
        non_basis: list[int],
        variable_to_constraint_index: dict[int, int],
    ) -> CostChange:
        """
        Computes the cost change effect for a basic variable given its corresponding row in the tableau and the indices of non-basic variables.

        :param variable: a basic variable
        :type variable: Variable
        :param non_basis: indices of non-basic variables
        :type non_basis: list[int]
        :param variable_to_constraint_index: a mapping from variable indices to their corresponding row indices in the tableau
        :type variable_to_constraint_index: dict[int, int]
        :return: cost change effect for the given basic variable
        :rtype: CostChange
        """
        # TODO: compute the cost change effect for a basic variable given its corresponding row in the tableau and the indices of non-basic variables
        #
        # 1) determine the row index in the tableau corresponding to the given basic variable using the `variable_to_constraint_index` mapping
        #    remember that the first row of the tableau corresponds to the objective
        # 2) extract the row of the tableau corresponding to the basic variable using the determined row index
        # 3) from this row, extract the coefficients corresponding to non-basic variables
        # 4) extract the reduced costs of non-basic variables from the objective coefficients using the indices of non-basic variables
        # 5) the variable's contstraint bound is equal to the coefficient scaling the variable's change in the objective
        # 6) compute the maximum allowed increase (delta_max) and decrease (delta_min) of the variable's cost coefficient while keeping the current basis optimal
        #    a) to compute delta_max, consider only the negative coefficients of non-basic variables in the variable's row
        #       and compute the minimum ratio of the negated reduced cost to the corresponding coefficient
        #    b) to compute delta_min, consider only the positive coefficients of non-basic variables in the variable's row
        #       and compute the maximum ratio of the negated reduced cost to the corresponding coefficient
        #    c) if there are no negative coefficients of non-basic variables in the variable's row, delta_max is infinite
        #    d) if there are no positive coefficients of non-basic variables in the variable's row, delta_min is negative infinite
        # 7) return a CostChange object with the computed objective delta coefficient, delta_min and delta_max

        row_index = variable_to_constraint_index[variable.index] + 1
        row = self.table[row_index]
        objective_row = self.table[0]

        delta_max_candidates = []
        delta_min_candidates = []

        for j in non_basis:
            coef = row[j]
            reduced_cost = objective_row[j]

            if coef < 0:
                delta_max_candidates.append(-reduced_cost / coef)

            elif coef > 0:
                delta_min_candidates.append(-reduced_cost / coef)

        delta_max = min(delta_max_candidates) if delta_max_candidates else float('inf')
        delta_min = max(delta_min_candidates) if delta_min_candidates else float('-inf')

        objective_delta_coefficient = row[-1]

        return CostChange(
            variable=variable,
            objective_delta_coefficient=objective_delta_coefficient,
            delta_min=delta_min,
            delta_max=delta_max
        )

        #raise NotImplementedError()

    def _compute_cost_change_out_of_basis(self, variable: Variable) -> CostChange:
        """
        Computes the cost change effect for a non-basic variable given its reduced cost in the tableau.

        :param variable: a non-basic variable
        :type variable: Variable
        :return: cost change effect for the given non-basic variable
        :rtype: CostChange
        """
        # TODO: compute the cost change effect for a non-basic variable given its reduced cost in the tableau
        #
        # 1) extract the reduced cost of the given non-basic variable from the objective coefficients using its index
        # 2) since the model is in the standard form, we can safely decrease the cost coefficient of this non-basic variable
        #    by any positive amount without losing optimality, so delta_min is negative infinite
        # 3) the maximum allowed increase (delta_max) of the variable's cost coefficient is equal to the reduced cost
        # 4) since the variable is non-basic, its change in the objective is equal to 0, as long as it's out of the basis
        # 5) return a CostChange object with the computed objective delta coefficient, delta_min and delta_max

        reduced_cost = self.table[0][variable.index]
        delta_min = float('-inf')
        delta_max = reduced_cost
        objective_delta_coefficient = 0.0

        return CostChange(
            variable=variable,
            objective_delta_coefficient=objective_delta_coefficient,
            delta_min=delta_min,
            delta_max=delta_max
        )

        #raise NotImplementedError()

    def __str__(self) -> str:
        def cell(x: str, w: int) -> str:
            return "{0: >{1}}".format(x, w)

        cost_name = self.model.objective.name()
        basis = self.extract_basis()
        header = (
            ["basis", cost_name] + [var.name for var in self.model.variables] + ["b"]
        )
        longest_col = max([len(h) for h in header])

        rows = [[cost_name]] + [[self.model.variables[i].name] for i in basis]

        for i, r in enumerate(rows):
            cost_factor = 0.0 if i > 0 else 1.0
            r += ["{:.3f}".format(v) for v in [cost_factor] + list(self.table[i])]
            longest_col = max(longest_col, max([len(v) for v in r]))

        header = [cell(h, longest_col) for h in header]
        rows = [[cell(v, longest_col) for v in row] for row in rows]

        cell_sep = " | "

        result = cell_sep.join(header) + "\n"
        for row in rows:
            result += cell_sep.join(row) + "\n"
        return result
