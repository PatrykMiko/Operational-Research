from dataclasses import dataclass

from saport.linear_programming.expressions.affine_substitution import AffineSubstitution
from saport.linear_programming.expressions.expression import Expression
from saport.linear_programming.expressions.variable import Variable
from saport.linear_programming.model import Model


@dataclass
class VariablePreprocessingResult:
    """
    A normalized model together with the information required to reconstruct
    assignments for the original model.
    """

    model: Model
    """normalized model used by downstream transformations or solvers"""

    original_model: Model
    """original user-defined model"""

    substitutions: dict[Variable, AffineSubstitution]
    """reconstruction formulas for original variables"""

    def restore_assignment(self, normalized_assignment: list[float]) -> list[float]:
        """
        Reconstructs an assignment aligned with the original model variables.

        :param normalized_assignment: assignment for the normalized model
        :type normalized_assignment: list[float]
        :return: assignment aligned with the original model variables
        :rtype: list[float]
        """
        return [
            self.substitutions[var].evaluate(normalized_assignment)
            for var in self.original_model.variables
        ]


def preprocess_variables(model: Model) -> VariablePreprocessingResult:
    """
    Creates a fresh model with normalized variables and the corresponding
    reconstruction substitutions for the original variables.

    :param model: user-defined model
    :type model: Model
    :return: variable preprocessing result
    :rtype: VariablePreprocessingResult
    """
    normalized_model = Model(model.name)
    substitutions = {
        var: _substitution_for_variable(var, normalized_model)
        for var in model.variables
    }

    _add_original_constraints(model, normalized_model, substitutions)
    _add_upper_bound_constraints(model, normalized_model, substitutions)
    normalized_model.objective = _substitute_objective(model, substitutions)

    return VariablePreprocessingResult(normalized_model, model, substitutions)


def _substitution_for_variable(
    variable: Variable, normalized_model: Model
) -> AffineSubstitution:
    lower_bound = variable.domain.lower_bound
    upper_bound = variable.domain.upper_bound

    if lower_bound is None and upper_bound is None:
        positive = normalized_model.create_variable(f"{variable.name}_extra_1")
        negative = normalized_model.create_variable(f"{variable.name}_extra_2")
        return AffineSubstitution(positive - negative)

    if lower_bound is None:
        assert upper_bound is not None
        extra_variable = normalized_model.create_variable(f"{variable.name}_extra")
        return AffineSubstitution(-extra_variable, upper_bound)

    normalized_variable_name = (
        variable.name if lower_bound == 0 else f"{variable.name}_extra"
    )
    normalized_variable = normalized_model.create_variable(normalized_variable_name)
    return AffineSubstitution(Expression(normalized_variable), lower_bound)


def _add_upper_bound_constraints(
    original_model: Model,
    normalized_model: Model,
    substitutions: dict[Variable, AffineSubstitution],
) -> None:
    for variable in original_model.variables:
        lower_bound = variable.domain.lower_bound
        upper_bound = variable.domain.upper_bound
        if lower_bound is None or upper_bound is None:
            continue

        substitution = substitutions[variable]
        normalized_model.add_constraint(
            substitution.expression <= upper_bound - substitution.offset
        )


def _add_original_constraints(
    original_model: Model,
    normalized_model: Model,
    substitutions: dict[Variable, AffineSubstitution],
) -> None:
    for constraint in original_model.constraints:
        normalized_model.add_constraint(constraint.substitute(substitutions))


def _substitute_objective(
    original_model: Model,
    substitutions: dict[Variable, AffineSubstitution],
):
    return original_model.objective.substitute(substitutions)
