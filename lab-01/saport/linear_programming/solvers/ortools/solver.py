import ortools.math_opt.python.mathopt as mathopt  # pyright: ignore[reportMissingTypeStubs]
from saport.linear_programming.result import Result
from saport.linear_programming.solution import Solution
from saport.linear_programming.solver import Solver
from saport.linear_programming.expressions import ConstraintType, ObjectiveType


class OrToolsSolver(Solver[Solution]):
    """
    A linear programming solver using the OR-Tools MathOpt library.
    Read more: https://developers.google.com/optimization/math_opt
    """

    def _create_ortools_model(
        self,
    ) -> tuple[mathopt.Model, dict[mathopt.Variable, int]]:
        """
        Translated a SAPORT model into a MathOpt model

        :return: MathOpt model and a mapping between the MathOpt and SAPORT variables
        :rtype: tuple[Model, dict[Variable, int]]
        """
        ortools_model = mathopt.Model(name=self.model.name)
        vars_mapping = {
            ortools_model.add_variable(lb=0, name=var.name): var.index
            for var in self.model.variables
        }

        for constraint in self.model.constraints:
            constr_coefficients = constraint.expression.get_coefficients(
                self.model.variables
            )
            constr_atoms = [
                var * constr_coefficients[index]
                for var, index in vars_mapping.items()
                if constr_coefficients[index] != 0
            ]
            constr_expression = sum(constr_atoms)

            match constraint.type:
                case ConstraintType.GE:
                    ortools_model.add_linear_constraint(  # pyright: ignore[reportUnknownMemberType]
                        constr_expression >= constraint.bound
                    )
                case ConstraintType.LE:
                    ortools_model.add_linear_constraint(  # pyright: ignore[reportUnknownMemberType]
                        constr_expression <= constraint.bound
                    )
                case ConstraintType.EQ:
                    ortools_model.add_linear_constraint(  # pyright: ignore[reportUnknownMemberType]
                        constr_expression == constraint.bound
                    )

        if self.model.objective.type == ObjectiveType.SAT:
            return ortools_model, vars_mapping

        obj_coefficients = self.model.objective.expression.get_coefficients(
            self.model.variables
        )
        obj_atoms = [
            var * obj_coefficients[index]
            for var, index in vars_mapping.items()
            if obj_coefficients[index] != 0
        ]
        obj_expression = sum(obj_atoms)

        match self.model.objective.type:
            case ObjectiveType.MAX:
                ortools_model.maximize_linear_objective(obj_expression)
            case ObjectiveType.MIN:
                ortools_model.minimize_linear_objective(obj_expression)

        return ortools_model, vars_mapping

    def _translate_solution(
        self,
        vars_values: dict[mathopt.Variable, float],
        vars_mapping: dict[mathopt.Variable, int],
    ) -> Solution:
        """
        Translates a MathOpt solution into a SAPORT one.

        :param vars_values: value assignment to the MathOpt variables
        :type vars_values: dict[mathopt.Variable, float]
        :param vars_mapping: mapping between MathOpt and SAPORT variables
        :type vars_mapping: dict[mathopt.Variable, int]
        :return: a SAPORT compatible solution
        :rtype: Solution
        """
        assignment = [0.0] * len(vars_mapping)
        for var, index in vars_mapping.items():
            assignment[index] = vars_values[var]
        return Solution(self.model, assignment)

    def solve(self) -> Result[Solution]:
        ortools_model, vars_mapping = self._create_ortools_model()
        params = mathopt.SolveParameters()
        result = mathopt.solve(ortools_model, mathopt.SolverType.GLOP, params=params)

        match result.termination.reason:
            case mathopt.TerminationReason.UNBOUNDED:
                return Result[Solution].unbounded()
            case mathopt.TerminationReason.INFEASIBLE:
                return Result[Solution].infeasible()
            case mathopt.TerminationReason.OPTIMAL:
                return Result[Solution].with_solution(
                    self._translate_solution(result.variable_values(), vars_mapping)
                )
            case _:
                return Result[Solution].errored()
