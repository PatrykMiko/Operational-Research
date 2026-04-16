import math

import numpy as np
import pytest

from saport.linear_programming.expressions import (
    Constraint,
    ConstraintType,
    Domain,
    Expression,
    ObjectiveType,
    Variable,
)
from saport.linear_programming.model import Model
from saport.linear_programming.solvers.simplex.cost_change import CostChange
from saport.linear_programming.solvers.simplex.solver import SimplexSolver
from saport.linear_programming.solvers.simplex.simplex_solution import SimplexSolution
from saport.linear_programming.solvers.simplex.tableau import Tableau
import saport.linear_programming.transformations.duality as duality


def assert_cost_change(
    cost_change: CostChange,
    *,
    variable_name: str,
    objective_delta_coefficient: float,
    delta_min: float,
    delta_max: float,
) -> None:
    assert cost_change.variable.name == variable_name
    assert math.isclose(
        cost_change.objective_delta_coefficient,
        objective_delta_coefficient,
        abs_tol=1e-6,
    )

    if math.isinf(delta_min):
        assert math.isinf(cost_change.delta_min) and cost_change.delta_min < 0
    else:
        assert math.isclose(cost_change.delta_min, delta_min, abs_tol=1e-6)

    if math.isinf(delta_max):
        assert math.isinf(cost_change.delta_max) and cost_change.delta_max > 0
    else:
        assert math.isclose(cost_change.delta_max, delta_max, abs_tol=1e-6)


def basis_model(variables_n: int) -> Model:
    model = Model("basis")
    for i in range(variables_n):
        model.create_variable(f"x{i}")
    return model


def baseline_nonnegative_model() -> Model:
    model = Model("baseline_nonnegative")
    x = model.create_variable("x")
    y = model.create_variable("y")

    model.add_constraint(x + y <= 4)
    model.add_constraint(x <= 2)
    model.add_constraint(y <= 3)
    model.maximize(3 * x + 2 * y)
    return model


def normalized_min_model() -> Model:
    model = Model("normalized_min")
    x = model.create_variable("x")
    y = model.create_variable("y")

    model.add_constraint(x + y >= 4)
    model.minimize(x + y)
    return model


def preprocessing_required_model() -> Model:
    model = Model("bounded_interval")
    x = model.create_variable("x", lb=5, ub=7)
    model.maximize(x)
    return model


def supported_dual_model() -> Model:
    model = Model("supported_dual")
    x = model.create_variable("x")
    y = model.create_variable("y")

    model.add_constraint(x + y <= 4)
    model.add_constraint(2 * x + y <= 5)
    model.maximize(3 * x + 2 * y)
    return model


def solve(model: Model):
    result = SimplexSolver(model).solve()
    assert result.solution is not None, "solver should return a solution"
    return result.solution


class TestDualityAndSensitivityOffline:
    @pytest.mark.parametrize(
        "objective_row, expected",
        [
            ([0.0, 0.0, 2.0, 2.0], True),
            ([0.0, 0.0, 0.0, 2.0], False),
        ],
    )
    def test_tableau_is_unique(self, objective_row, expected):
        model = basis_model(4)
        tableau = Tableau(
            model,
            np.array(
                [
                    objective_row + [0.0],
                    [1.0, 0.0, 1.0, 0.0, 3.0],
                    [0.0, 1.0, -1.0, 0.0, 2.0],
                ]
            ),
        )
        dummy_assignment = [0.0] * len(model.variables)
        solution = SimplexSolution(model, dummy_assignment, tableau, tableau)

        assert solution.is_unique() == expected

    def test_tableau_is_unique_raises_for_nonoptimal_tableau(self):
        model = basis_model(4)
        tableau = Tableau(
            model,
            np.array(
                [
                    [0.0, 0.0, -1.0, 1.0, 0.0],
                    [1.0, 0.0, 1.0, 0.0, 3.0],
                    [0.0, 1.0, -1.0, 0.0, 2.0],
                ]
            ),
        )
        dummy_assignment = [0.0] * len(model.variables)
        solution = SimplexSolution(model, dummy_assignment, tableau, tableau)

        with pytest.raises(ValueError):
            solution.is_unique()

    def test_tableau_computes_cost_changes(self):
        solution = solve(baseline_nonnegative_model())
        changes = solution.tableau.compute_cost_changes()
        expected_changes = [
            ("x", 2.0, -1.0, math.inf),
            ("y", 2.0, -2.0, 1.0),
            ("s0", 0.0, -math.inf, 2.0),
            ("s1", 0.0, -math.inf, 1.0),
            ("s2", 1.0, -1.0, 2.0),
        ]

        assert len(changes) == len(expected_changes)
        for change, (
            variable_name,
            objective_delta_coefficient,
            delta_min,
            delta_max,
        ) in zip(changes, expected_changes, strict=True):
            assert_cost_change(
                change,
                variable_name=variable_name,
                objective_delta_coefficient=objective_delta_coefficient,
                delta_min=delta_min,
                delta_max=delta_max,
            )

    def test_tableau_compute_cost_changes_raises_on_incomplete_basis(self):
        model = basis_model(2)
        tableau = Tableau(
            model,
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [2.0, 0.0, 1.0],
                ]
            ),
        )

        with pytest.raises(ValueError):
            tableau.compute_cost_changes()

    def test_tableau_properly_computes_cost_change_in_basis(self):
        solution = solve(baseline_nonnegative_model())
        tableau = solution.tableau
        basis = tableau.extract_basis()
        non_basis = tableau.extract_non_basis()
        variable_to_constraint_index = {
            variable_index: row_index for row_index, variable_index in enumerate(basis)
        }

        x_change = tableau._compute_cost_change_in_basis(
            tableau.model.variables[0], non_basis, variable_to_constraint_index
        )
        y_change = tableau._compute_cost_change_in_basis(
            tableau.model.variables[1], non_basis, variable_to_constraint_index
        )

        assert_cost_change(
            x_change,
            variable_name="x",
            objective_delta_coefficient=2.0,
            delta_min=-1.0,
            delta_max=math.inf,
        )
        assert_cost_change(
            y_change,
            variable_name="y",
            objective_delta_coefficient=2.0,
            delta_min=-2.0,
            delta_max=1.0,
        )

    def test_tableau_properly_computes_cost_change_out_of_basis(self):
        solution = solve(baseline_nonnegative_model())
        tableau = solution.tableau
        slack_change = tableau._compute_cost_change_out_of_basis(
            tableau.model.variables[2]
        )

        assert_cost_change(
            slack_change,
            variable_name="s0",
            objective_delta_coefficient=0.0,
            delta_min=-math.inf,
            delta_max=2.0,
        )

    @pytest.mark.parametrize(
        "objective_type, expected_min, expected_max",
        [
            (ObjectiveType.MAX, -2.0, 5.0),
            (ObjectiveType.MIN, -5.0, 2.0),
        ],
    )
    def test_simplex_solution_maps_cost_change_to_original_objective(
        self, objective_type, expected_min, expected_max
    ):
        model = baseline_nonnegative_model()
        if objective_type == ObjectiveType.MIN:
            model.minimize(model.objective.expression)
        solution = solve(model)
        variable = solution.model.variables[0]
        original_cost_change = solution._cost_change_for_original_objective(
            variable, CostChange(variable, 3.0, -2.0, 5.0)
        )

        assert_cost_change(
            original_cost_change,
            variable_name=variable.name,
            objective_delta_coefficient=3.0,
            delta_min=expected_min,
            delta_max=expected_max,
        )

    def test_simplex_solution_computes_cost_changes_for_normalized_max_model(self):
        solution = solve(baseline_nonnegative_model())
        tableau_changes = solution.tableau.compute_cost_changes()
        solution_changes = solution.compute_cost_changes()

        assert len(solution_changes) == len(solution.model.variables)
        for variable, solution_change, tableau_change in zip(
            solution.model.variables, solution_changes, tableau_changes[:2], strict=True
        ):
            assert solution_change.variable == variable
            assert math.isclose(
                solution_change.objective_delta_coefficient,
                tableau_change.objective_delta_coefficient,
                abs_tol=1e-6,
            )
            assert math.isclose(
                solution_change.delta_min, tableau_change.delta_min, abs_tol=1e-6
            )
            assert solution_change.delta_max == tableau_change.delta_max

    def test_simplex_solution_computes_cost_changes_for_normalized_min_model(self):
        solution = solve(normalized_min_model())
        tableau_changes = solution.tableau.compute_cost_changes()[
            :len(solution.model.variables)
        ]
        solution_changes = solution.compute_cost_changes()

        for solution_change, tableau_change in zip(
            solution_changes, tableau_changes, strict=True
        ):
            assert math.isclose(
                solution_change.objective_delta_coefficient,
                tableau_change.objective_delta_coefficient,
                abs_tol=1e-6,
            )
            assert math.isclose(
                solution_change.delta_min, -tableau_change.delta_max, abs_tol=1e-6
            )
            if math.isinf(tableau_change.delta_min):
                assert math.isinf(solution_change.delta_max)
            else:
                assert math.isclose(
                    solution_change.delta_max, -tableau_change.delta_min, abs_tol=1e-6
                )

    def test_simplex_solution_compute_cost_changes_raises_when_preprocessing_is_required(
        self,
    ):
        solution = solve(preprocessing_required_model())

        with pytest.raises(ValueError):
            solution.compute_cost_changes()

    def test_create_dual_builds_expected_model(self):
        primal = supported_dual_model()
        dual = duality.create_dual(primal)

        assert dual.objective.type == ObjectiveType.MIN
        assert len(dual.variables) == len(primal.constraints)
        assert len(dual.constraints) == len(primal.variables)
        assert dual.variables[0].domain == Domain(0, None)
        assert dual.variables[1].domain == Domain(0, None)
        assert dual.constraints[0].type == ConstraintType.GE
        assert dual.constraints[1].type == ConstraintType.GE

    def test_create_dual_solves_to_matching_objective(self):
        primal = supported_dual_model()
        dual = duality.create_dual(primal)

        primal_solution = solve(primal)
        dual_solution = solve(dual)

        assert math.isclose(
            primal_solution.objective_value(),
            dual_solution.objective_value(),
            abs_tol=1e-7,
        )

    def test_create_dual_respects_allow_preprocessing_flag(self):
        model = preprocessing_required_model()

        with pytest.raises(ValueError):
            duality.create_dual(model, allow_preprocessing=False)

        dual = duality.create_dual(model, allow_preprocessing=True)
        result = SimplexSolver(dual).solve()
        assert result.solution is not None

    def test_duality_builds_dual_objective(self):
        primal = supported_dual_model()
        dual = Model("dual")
        dual_variables = [
            duality._create_dual_variable(dual, constraint)
            for constraint in primal.constraints
        ]

        objective = duality._build_dual_objective(primal, dual_variables)
        coefficients = objective.expression.get_coefficients(dual_variables)

        assert objective.type == ObjectiveType.MIN
        assert coefficients == [4.0, 5.0]

    def test_duality_builds_dual_constraint(self):
        primal = supported_dual_model()
        dual = Model("dual")
        dual_variables = [
            duality._create_dual_variable(dual, constraint)
            for constraint in primal.constraints
        ]

        x_constraint = duality._build_dual_constraint(
            primal, primal.variables[0], dual_variables
        )
        y_constraint = duality._build_dual_constraint(
            primal, primal.variables[1], dual_variables
        )

        assert x_constraint.expression.get_coefficients(dual_variables) == [1.0, 2.0]
        assert x_constraint.bound == 3.0
        assert x_constraint.type == ConstraintType.GE

        assert y_constraint.expression.get_coefficients(dual_variables) == [1.0, 1.0]
        assert y_constraint.bound == 2.0
        assert y_constraint.type == ConstraintType.GE
