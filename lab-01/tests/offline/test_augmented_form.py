from typing import List, Tuple
from saport.linear_programming.model import Model
from saport.linear_programming.solvers.simplex.solver import SimplexSolver
from saport.linear_programming.solvers.simplex.tableau import Tableau
from saport.linear_programming.expressions import Variable, Constraint, ConstraintType, ObjectiveType
from copy import deepcopy
import numpy as np
from tests.offline.cases import model_1, model_2, augmented_1, augmented_2, tableau_1, tableau_2
import pytest


class TestAugmentation:
    @pytest.mark.parametrize(
        "input,expected", [(model_1(), augmented_1()), (model_2(), augmented_2())]
    )
    def test_augment_model(self, input: Model, expected: Model):
        solver = SimplexSolver(input)
        result = solver._augment_model(input)
        result_constr_n = len(result.constraints)
        expect_constr_n = len(expected.constraints)
        assert result_constr_n == expect_constr_n, (
            f"number of constraints should not be changed - got {result_constr_n}, expected: {expect_constr_n}"
        )

        result_var_n = len(result.variables)
        expect_var_n = len(expected.variables)
        assert result_var_n == expect_var_n, (
            f"augmentation should add correct amount of slack/surplus variables - got {result_var_n}, expected: {expect_var_n}"
        )
        assert result.objective.type == ObjectiveType.MAX, (
            f"augmentation should modify the objective to be of type 'max'"
        )
        assert expected.objective.expression.get_coefficients(
            expected.variables
        ) == result.objective.expression.get_coefficients(expected.variables), (
            f"objective is not augmented correctly - got: {result.objective.expression}, expected {expected.objective.expression}"
        )
        for c in result.constraints:
            assert c.bound >= 0, (
                f"augmented model has only non-negative bounds - got constraint {c}"
            )
            assert c.type == ConstraintType.EQ, (
                f"augmented constraints are always of type '==' - got constraint {c}"
            )

            def sorted_coeffs(c: Constraint) -> list[float]:
                input_var_n = len(input.variables)
                coeffs = c.expression.get_coefficients(expected.variables)
                return coeffs[:input_var_n] + sorted(coeffs[input_var_n:])

            assert any(
                [sorted_coeffs(ec) == sorted_coeffs(c) for ec in expected.constraints]
            ), (
                f"constraint {c} is not augmented correctly, there is no constraint"
            )

    @pytest.mark.parametrize(
        "input,expected", [(augmented_1(), tableau_1()), (augmented_2(), tableau_2())]
    )
    def test_basic_initial_tableau(self, input: Model, expected: Tableau):
        solver = SimplexSolver(input)
        result = solver._basic_initial_tableau(input)
        assert result.table is not None, (
            f"the initial table in `Tableau` should not be `None` for model:\n{input}"
        )
        assert result.table.shape == expected.table.shape, (
            f"the initial tableau has incorrect shape - got: {result.table.shape}, expected: {expected.table.shape} for model:\n{input}"
        )
        assert np.allclose(result.table, expected.table), (
            f"the initial tablaux is incorrect - got:\n{result.table}expected:\n{expected.table} for model:\n{input}"
        )
