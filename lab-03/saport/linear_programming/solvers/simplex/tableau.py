from dataclasses import dataclass
from numpy.typing import NDArray
import numpy as np
import math
from saport.linear_programming.model import Model
from saport.linear_programming.solvers.simplex.constants import eps


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
        masked_coeffs = np.ma.masked_less_equal(
            coeffs, eps
        )
        bound_col = self.table[1:, -1]
        ratios = bound_col / masked_coeffs

        # +1 because we ignore the first row of the tableau, corresponding to the objective
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

        # make the pivot element equal to 1 by dividing the whole row by the pivot factor
        self.table[row] /= pivot_factor

        # we need to change all rows except the pivot row
        indices_to_change = np.arange(rows_n) != row

        # for each correct row we prepare a row that when subtracted will result in the column `col` having 0 coefficient
        mult_rows = self.table[indices_to_change, col, np.newaxis] * self.table[row, np.newaxis]

        # and we subtract them, which results in the column `col` having 0 coefficient in all rows except the pivot row, where it is 1
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
