from saport.linear_programming.solvers.simplex.constants import eps
from saport.linear_programming.solvers.simplex.tableau import Tableau
import numpy as np
import pytest


class TestSimplexOffline:
    @pytest.mark.parametrize(
        "objective_row, is_optimal",
        [
            ([3, 3], True),
            ([0, 0, 0], True),
            ([6, 2, -1], True),
            ([1, 3, -np.inf], True),
            ([3, -1, -np.inf], False),
            ([-2, -3, -1], False),
            ([-3, -np.inf, 1], False),
            ([-eps, -eps, -3], True),
        ],
    )
    def test_tableau_properly_determines_optimal_solutions(
        self, objective_row, is_optimal
    ):
        tableau = Tableau(None, np.array([objective_row, objective_row]))

        assert tableau.is_optimal() == is_optimal, (
            "tableau is unable to correctly determine if a solution is optimal:"
            + f"\n- for top row {objective_row} it should've returned {is_optimal}, it has not :("
        )

    @pytest.mark.parametrize(
        "objective_row, expected_col",
        [
            ([-np.inf, -5], 0),
            ([-1, -np.inf, 1], 1),
            ([np.inf, -eps, -1], 1),
            ([0, 0, -5, -np.inf], 2),
            ([-1, -5, -8, -1], 2),
        ],
    )
    def test_tableau_properly_chooses_entering_variable(
        self, objective_row, expected_col
    ):
        tableau = Tableau(None, np.array([objective_row, objective_row]))

        got = tableau.choose_entering_variable()
        assert got == expected_col, (
            "tableau is unable to choose entering variable properly:"
            + f"\n - for top row {objective_row} it should've returned {expected_col}, instead it produced {got}"
        )

    @pytest.mark.parametrize(
        "pivot_col, is_unbounded",
        [
            ([1, 2, 3], False),
            ([-1, 2, 3], False),
            ([1, -2, 3], False),
            ([1, 2, -3], False),
            ([-1, -2, 3], False),
            ([1, -2, -3], True),
            ([-1, -2, -3], True),
        ],
    )
    def test_tableau_properly_checking_if_unbounded(self, pivot_col, is_unbounded):

        tableau = Tableau(None, np.array([pivot_col]).T)

        assert tableau.is_unbounded(0) == is_unbounded, (
            "tableau is unable to detect unbounded problem:"
            + f"\n- for column with coefficients {pivot_col} it should've returned {is_unbounded}, it has not :("
        )

    @pytest.mark.parametrize(
        "table, expected_rows",
        [
            ([[0, 0, 0, 0], [0, 4, -2, 1], [0, 8, 501, 18]], [0, 1, 2]),
            ([[0, 0, 0, 0], [-2, 4, 6, 0], [0, -1, 0, 0]], [0, 1, 1]),
            ([[0, 0, 0, 0], [-10, 0, -2, 0.5], [8, 0, 8, 1]], [2, 0, 2]),
            ([[0, 0, 0, 0], [0, 4, -4, 2], [0, 8, 2, 1e22]], [0, 1, 2]),
            ([[1, 1, 1, 1], [0, 2, 3, 7], [0, 1, 2, 7]], [0, 1, 1]),
        ],
    )
    def test_tableau_properly_chooses_leaving_variable(self, table, expected_rows):
        tableau = Tableau(None, np.array(table))
        _, cols_n = tableau.table.shape

        for i in range(cols_n - 1):
            if tableau.table[1:, i].max() <= 0:
                # because there is no defined result when problem is unbounded
                continue

            got = tableau.choose_leaving_variable(i)
            assert got == expected_rows[i], (
                "tableau is unable to choose leaving variable properly:"
                + f"\n- for column {list(tableau.table[:, i])} and bounds column {list(tableau.table[:, -1])} it should've returned {expected_rows[i]}, instead it has produced {got}"
            )

    @pytest.mark.parametrize(
        "table_before, piv_row, piv_col, table_after",
        [
            (
                [
                    [0.0, -1.0, -2.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                    [0.0, -1.0, 1.0, 0.0, 1.0, 0.0, 2],
                    [-1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 3.0],
                ],
                2,
                2,
                [
                    [0.0, -3.0, 0.0, 0.0, 2.0, 0.0, 4.0],
                    [1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                    [0.0, -1.0, 1.0, 0.0, 1.0, 0.0, 2.0],
                    [-1.0, 1.0, 0.0, 0.0, -1.0, 1.0, 1.0],
                ],
            ),
            (
                [
                    [0.0, -4.0, -2.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                    [0.0, -1.0, 1.0, 0.0, 1.0, 0.0, 2.0],
                    [-1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 3.0],
                ],
                1,
                1,
                [
                    [4.0, 0.0, -2.0, 4.0, 0.0, 0.0, 4.0],
                    [1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 3.0],
                    [-1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 3.0],
                ],
            ),
            (
                [
                    [0.0, -3.0, -4.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0],
                    [-1.0, -1.0, 1.0, 0.0, 1.0, 0.0, 3.0],
                    [-1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 5.0],
                ],
                1,
                2,
                [
                    [4.0, 1.0, 0.0, 4.0, 0.0, 0.0, 4.0],
                    [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0],
                    [-2.0, -2.0, 0.0, -1.0, 1.0, 0.0, 2.0],
                    [-2.0, 0.0, 0.0, -1.0, 0.0, 1.0, 4.0],
                ],
            ),
        ],
    )
    def test_tableau_properly_pivots(self, table_before, piv_row, piv_col, table_after):
        table_before = np.array(table_before)
        table_after = np.array(table_after)
        tableau = Tableau(None, table_before)
        tableau.pivot(piv_row, piv_col)

        assert np.array_equal(tableau.table, table_after), (
            "tableau incorrectly pivots a table:"
            + f"\n- init table:\n{table_before}"
            + f"\n- pivot coords {(piv_row, piv_col)}"
            + f"\n- expected result:\n{table_after}"
            + f"\n- got:\n{tableau.table}"
        )
