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
        # TODO:
        # the solution is optimal when no variable can improve the objective
        # which happens when all coefficients in the objective row are non-negative.
        #
        # return `True` if all coefficients in the objective row are >= 0, `False` otherwise
        # tip 0. use the `objective_coefficients()`
        # tip 1. check the `eps` constant at the top of this file - use it as a tolerance

        for cof in  self.objective_coefficients():
            if cof < -eps:
                return False
        return True

        # raise NotImplementedError()

    def choose_entering_variable(self) -> TableauIndex:
        """
        Finds an index of the variable that should enter the basis.

        :return: an index of the entering variable (column)
        :rtype: TableauIndex
        """
        # TODO:
        # we pick the variable that can improve the objective the most, the one
        # with the most negative coefficient in the objective row.
        #
        # return the column index with the smallest coefficient in the objective row
        # tip. use the `objective_coefficients()`

        return self.objective_coefficients().argmin()



        # raise NotImplementedError()

    def is_unbounded(self, col: TableauIndex) -> bool:
        """
        Checks, whether the current tableau represent an unbounded problem
        given the selected entering variable.

        :param col: index of an entering variable
        :type col: TableauIndex
        :return: `True` if the problem is unbounded, `False` otherwise
        :rtype: bool
        """
        # TODO:
        # if no constraint limits how far the entering variable can grow, the objective
        # is unbounded and we can increase it without limit.
        #
        # return `True` if all coefficients in the constraint rows (excluding the objective row) 
        # of the specified column are <= 0, `False` otherwise

        constraint_coeffs = self.table[1:, col]
        for cof in constraint_coeffs:
            if cof > eps:
                return False
        return True

        #raise NotImplementedError()

    def choose_leaving_variable(self, col: TableauIndex) -> TableauIndex:
        """
        Finds an index of the variable that should leave the basis given
        the entering variable index.

        :param col: index of the entering variable
        :type col: TableauIndex
        :return: an index of the leaving variable (column)
        :rtype: TableauIndex
        """

        # TODO:
        # the leaving variable is the one that first hits zero as the entering variable increases.
        #
        # return the row index associated with the leaving variable:
        # 1) for each constraint row (not the objective row), compute the ratio:
        #    bound (last column) / coefficient in the entering variable's column
        # 2) only consider rows where the column coefficient is strictly positive
        #    (ignore rows with zero or negative coefficients - they don't limit growth)
        # 3) pick the row with the smallest such ratio
        # tip: remember that constraint rows start at index 1 in the table (row 0 is the objective)
        # tip: take care to not divide by 0 :)

        constraints = self.table[1:,:]
        active_constraints = [(i, c) for i, c in enumerate(constraints, start=1) if c[col]>eps]
        return min(active_constraints, key=lambda c: c[1][-1]/c[1][col])[0]

        #raise NotImplementedError()

    def pivot(self, row: TableauIndex, col: TableauIndex) -> None:
        """
        Performs a pivot operation given the coordinates of the entering and leaving variables.
        It modifies the tableau in-place and doesn't return anything.

        :param row: the row-coordinate represents the leaving variable
        :type row: TableauIndex
        :param col: the col-coordinate represents the entering variable
        :type col: TableauIndex
        """
        # TODO:
        # pivoting is the core of the simplex method - it swaps one variable into the basis
        # and another out, moving to a new vertex of the feasible region.
        #
        # pivot operation should transform the tableau to a form, where pivot column ('col')
        # contains only 0's with the exception of 1 in the pivot row ('row'), i.e.
        #
        #              col
        #       _ _ _ _ 0 _
        #       _ _ _ _ 0 _
        #  row  _ _ _ _ 1 _
        #       _ _ _ _ 0 _
        #
        # To achieve this goal, one has to transform the tableau in a way preserving the set of solutions
        # (remember, that tableau represents a set of linear equations, we don't want to break them!).
        # Therefore one can only use the following operations taught in secondary school:
        # - multiply the row (coefficients in the equation) by a scalar, e.g.
        #       4x + 5y = 4 | * 1/5 -> 4/5x + y = 4/5
        # - add one equation (optionally multiplied by a scalar) to another, e.g.
        #       4x - 3y = 7
        #       2x - 1y = 3
        #       ___________ -2*
        #       0x - 1y = 1
        #
        # In other words, one can only multiply the rows of the tableau by a scalar (numpy rows
        # can be easily multiplied), or add one row (possibly multiplied by a scalar) to another
        # (again, numpy supports this out of the box). There exists a fixed set of such operations
        # leading to the correct pivot.

        pivot_val = self.table[row, col]
        self.table[row, :] = self.table[row, :] / pivot_val

        for i in range(self.table.shape[0]):
            if i != row:
                factor = self.table[i, col]
                self.table[i, :] = self.table[i, :] - factor * self.table[row, :]

        #raise NotImplementedError()

    def extract_assignment(self) -> list[float]:
        """
        Extract a value assignment represented by the current state of tableau.

        :return: a list of values assigned to the variables in the tableau
        :rtype: list[int]
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
        :rtype: list[float]
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
