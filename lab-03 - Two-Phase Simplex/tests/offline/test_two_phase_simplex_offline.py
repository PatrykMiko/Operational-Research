from copy import deepcopy

import numpy as np
from saport.linear_programming.model import Model
from saport.linear_programming.solvers.simplex.solver import SimplexSolver
from saport.linear_programming.solvers.simplex.tableau import Tableau
import pytest
from conftest import indented_string


def model_example_solvable():
    model = Model("example_solvable")
    x1 = model.create_variable("x1")
    x2 = model.create_variable("x2")
    model.add_constraint(2 * x1 - x2 <= 100)
    model.add_constraint(2 * x1 - x2 <= -1)
    model.add_constraint(x1 + x2 <= 3)
    model.add_constraint(x1 + 2 * x2 >= 4)
    model.maximize(x1 + 2 * x2)
    return model


def model_example_infeasible():
    model = Model("example_infeasible")
    x1 = model.create_variable("x1")
    x2 = model.create_variable("x2")
    model.add_constraint(x1 + x2 <= 3)
    model.add_constraint(x1 + x2 >= 4)
    model.maximize(x1 + 3 * x2)
    return model


def presolve_model_example_solvable():
    model = model_example_solvable()
    solver = SimplexSolver(model)
    model = solver._augment_model(model)
    r1 = model.create_variable("R1")
    r3 = model.create_variable("R3")
    model.constraints[1].expression += r1
    model.constraints[3].expression += r3
    model.maximize(-r1 - r3)
    solver._artificial = {r1: model.constraints[1], r3: model.constraints[3]}
    return solver, model


def presolve_model_example_infeasible():
    model = model_example_infeasible()
    solver = SimplexSolver(model)
    model = solver._augment_model(model)
    r1 = model.create_variable("R1")
    model.constraints[1].expression += r1
    model.maximize(-r1)
    solver._artificial = {r1: model.constraints[1]}
    return solver, model


class TestTwoPhaseSimplexOffline:
    @pytest.mark.parametrize(
        "model, art_n", [(model_example_solvable(), 2), (model_example_infeasible(), 1)]
    )
    def test_solver_properly_adds_artificial_variables(self, model, art_n):
        solver = SimplexSolver(model)
        augmented_model = solver._augment_model(model)
        art_model = deepcopy(augmented_model)
        art_vars = solver._add_artificial_variables(art_model)

        assert len(art_vars) == art_n, (
            "incorrect number of generated artificial variables:"
            + f"\n- expected: {art_n}"
            + f"\n- got: {len(art_vars)}"
            + f"\n- for augmented model:\n{indented_string(str(augmented_model))}"
            + f"\n- got model with art. vars: {art_model}"
        )

        for _, art_constraint in art_vars.items():
            for _, slack_constraint in solver._slacks.items():
                assert art_constraint.index != slack_constraint.index, (
                    "some artificial variables were added to constraints with slacks:"
                    + f"\n- augmented model:\n{augmented_model}"
                    + f"\n- got model with art. vars:\n{art_vars}"
                )

        art_indexes = [c.index for c in art_vars.values()]
        assert len(art_indexes) == len(set(art_indexes)), (
            "some artificial variables have been added to the same constraint!"
            + f"\n- augmented model:\n{augmented_model}"
            + f"\n- got model with art. vars:\n{art_vars}"
        )

    @pytest.mark.parametrize(
        "presolved_solver_and_model, expected_tableau",
        [
            (
                presolve_model_example_solvable(),
                [
                    [1.0, -3.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, -5.0],
                    [2.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100],
                    [-2.0, 1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 1],
                    [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 3],
                    [1.0, 2.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 4],
                ],
            ),
            (
                presolve_model_example_infeasible(),
                [
                    [-1.0, -1.0, 0.0, 1.0, 0.0, -4.0],
                    [1.0, 1.0, 1.0, 0.0, 0.0, 3.0],
                    [1.0, 1.0, 0.0, -1.0, 1.0, 4.0],
                ],
            ),
        ],
    )
    def test_solver_properly_presolving_initial_tableau(
        self, presolved_solver_and_model, expected_tableau
    ):
        solver, model = presolved_solver_and_model
        tableau = solver._presolve_initial_tableau(model)
        expected_tableau = np.array(expected_tableau)
        assert np.array_equal(tableau.table, np.array(expected_tableau)), (
            "solver does not initialize `presolve` tableau properly:"
            + f"\n- expected:\n{indented_string(str(expected_tableau))}"
            + f"\n- got:\n{indented_string(str(tableau.table))}"
            + f"\n- model:\n{indented_string(str(model))}"
        )

    @pytest.mark.parametrize(
        "presolved_solver_and_model, expected, table",
        [
            (
                presolve_model_example_solvable(),
                False,
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 101.0],
                    [
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        -0.19999999999999996,
                        -0.4,
                        0.19999999999999996,
                        0.4,
                        1.8,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        -0.20000000000000018,
                        0.6000000000000001,
                        0.20000000000000018,
                        -0.6000000000000001,
                        0.7999999999999998,
                    ],
                    [1.0, 0.0, 0.0, 0.0, 0.4, -0.2, -0.4, 0.2, 0.4],
                ],
            ),
            (
                presolve_model_example_infeasible(),
                True,
                [
                    [0.0, 0.0, 1.0, 1.0, 0.0, -1.0],
                    [1.0, 1.0, 1.0, 0.0, 0.0, 3.0],
                    [0.0, 0.0, -1.0, -1.0, 1.0, 1.0],
                ],
            ),
        ],
    )
    def test_solver_properly_checking_if_art_vars_positive(
        self, presolved_solver_and_model, expected, table
    ):
        solver, model = presolved_solver_and_model
        table = np.array(table)
        tableau = Tableau(model, table)
        result = solver._artificial_variables_are_positive(tableau)

        assert result == expected, (
            "solver does not properly check whether any artificial variables are positive:"
            + f"\n- expected: {expected}"
            + f"\n- got: {result}"
            + f"\n- for tableauax:\n{indented_string(str(tableau))}"
        )

    @pytest.mark.parametrize(
        "init_model, presolved_solver_and_model, optimal_presolve_table, expected_restored_table",
        [
            (
                model_example_solvable(),
                presolve_model_example_solvable(),
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 101.0],
                    [
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        -0.19999999999999996,
                        -0.4,
                        0.19999999999999996,
                        0.4,
                        1.8,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        -0.20000000000000018,
                        0.6000000000000001,
                        0.20000000000000018,
                        -0.6000000000000001,
                        0.7999999999999998,
                    ],
                    [1.0, 0.0, 0.0, 0.0, 0.4, -0.2, -0.4, 0.2, 0.4],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0, 1.1102230246251565e-16, -1.0, 4.0],
                    [0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 101.0],
                    [0.0, 1.0, 0.0, 0.0, -0.19999999999999996, -0.4, 1.8],
                    [
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        -0.20000000000000018,
                        0.6000000000000001,
                        0.7999999999999998,
                    ],
                    [1.0, 0.0, 0.0, 0.0, 0.4, -0.2, 0.4],
                ],
            ),
        ],
    )
    def test_solver_properly_restores_initial_tableau(
        self,
        init_model,
        presolved_solver_and_model,
        optimal_presolve_table,
        expected_restored_table,
    ):
        augmented_model = SimplexSolver(init_model)._augment_model(init_model)
        solver, model = presolved_solver_and_model
        table = np.array(optimal_presolve_table)
        tableau = Tableau(model, table)
        expected_restored_table = np.array(expected_restored_table)
        restored_tableau = solver._restore_initial_tableau(
            deepcopy(tableau), augmented_model
        )
        assert np.array_equal(expected_restored_table, restored_tableau.table), (
            "solver does not restore initial tableau properly:"
            + f"\n- expected:\n{indented_string(str(expected_restored_table))}"
            + f"\n- got:\n{indented_string(str(restored_tableau.table))}"
            + f"\n- input tableau:\n{indented_string(str(tableau))}"
            + f"\n- input model:\n{indented_string(str(augmented_model))}"
        )
