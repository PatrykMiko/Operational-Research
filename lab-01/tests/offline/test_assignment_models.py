import math
from typing import Callable

from saport.linear_programming.model import Model
from saport.linear_programming.solvers.ortools.solver import OrToolsSolver
from assignment import assignment_1, assignment_2, assignment_3, assignment_4


class TestAssignmentModels:

    def test_assignment_1_is_correctly_modelled(self):
        self.assert_model_results(assignment_1, 80.0)

    def test_assignment_2_is_correctly_modelled(self):
        self.assert_model_results(assignment_2, 10800.0)

    def test_assignment_3_is_correctly_modelled(self):
        self.assert_model_results(assignment_3, 21.818181818)

    def test_assignment_4_is_correctly_modelled(self):
        self.assert_model_results(assignment_4, 4000.0)

    def assert_model_results(self, model_constructor: Callable[[], Model], expected_objective: float) -> None:
        model = model_constructor()
        solver = OrToolsSolver(model)
        result = solver.solve()

        assert result.solution is not None, "the solver failed to solve the model"

        got_objective = result.solution.objective_value()
        assert math.isclose(got_objective, expected_objective, abs_tol=0.0000001), (
            f"`{model_constructor.__name__}`: got objective `{got_objective}`, expected `{expected_objective}`"
        )
