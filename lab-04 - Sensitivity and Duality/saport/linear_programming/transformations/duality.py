from typing import Optional

from saport.linear_programming.expressions import (
    Constraint,
    ConstraintType,
    DomainSign,
    Expression,
    Objective,
    ObjectiveType,
    Variable,
)
from saport.linear_programming.model import Model
from saport.linear_programming.transformations.variable_preprocessing import (
    preprocess_variables,
)


def create_dual(model: Model, allow_preprocessing: bool = False) -> Model:
    """
    Returns the dual of the model.

    :param model: primal model
    :type model: Model
    :param allow_preprocessing: whether to preprocess variable domains first
    :type allow_preprocessing: bool
    :return: a new model representing the dual of the input model
    :rtype: Model
    """
    # TODO: create the dual of the model
    #
    # 1) if `allow_preprocessing` is `True`, preprocess variable domains using `preprocess_variables()`, otherwise use the original model,
    #    this is our primal model for the rest of the function
    # 2) validate that the primal model can be dualized using `_validate_dualizable()`
    # 3) create a new model for the dual, with a name derived from the primal model name, e.g. `f"{primal.name}_dual"`
    # 4) create dual variables for each primal constraint using `_create_dual_variable()`, store them in a list
    # 5) build the dual objective using `_build_dual_objective()`, set it as the dual model objective
    # 6) for each primal variable, build a dual constraint using `_build_dual_constraint()`, add it to the dual model
    # 7) return the dual model

    primal = preprocess_variables(model).model if allow_preprocessing else model
    _validate_dualizable(primal)
    dual_model = Model(name=f"{primal.name}_dual")
    dual_variables = []
    for constraint in primal.constraints:
        dual_var = _create_dual_variable(dual_model, constraint)
        dual_variables.append(dual_var)

    dual_model.objective = _build_dual_objective(primal, dual_variables)
    for variable in primal.variables:
        dual_constraint = _build_dual_constraint(primal, variable, dual_variables)
        dual_model.add_constraint(dual_constraint)

    return dual_model

    #raise NotImplementedError()


def _validate_dualizable(model: Model) -> None:
    if model.objective.type != ObjectiveType.MAX:
        raise ValueError("Dual is only implemented for maximization problems")

    for variable in model.variables:
        if variable.domain.sign == DomainSign.OTHER:
            raise ValueError(
                "Dual is only implemented for variables with domains "
                "(-inf, 0], [0, inf), or (-inf, inf): "
                f"{variable.name} has domain {variable.domain}"
            )


def _dual_variable_bounds(
    constraint: Constraint,
) -> tuple[Optional[float], Optional[float]]:
    match constraint.type:
        case ConstraintType.GE:
            return (None, 0)
        case ConstraintType.LE:
            return (0, None)
        case ConstraintType.EQ:
            return (None, None)
        case _:
            raise ValueError(
                f"Unknown constraint type: {constraint.type} for constraint {constraint}"
            )

    raise NotImplementedError()


def _create_dual_variable(dual: Model, constraint: Constraint) -> Variable:
    lb, ub = _dual_variable_bounds(constraint)
    return dual.create_variable(f"y{constraint.index}", lb=lb, ub=ub)


def _dual_constraint_type(variable: Variable) -> ConstraintType:
    match variable.domain.sign:
        case DomainSign.NONPOSITIVE:
            return ConstraintType.LE
        case DomainSign.NONNEGATIVE:
            return ConstraintType.GE
        case DomainSign.FREE:
            return ConstraintType.EQ
        case DomainSign.OTHER:
            raise ValueError(f"Invalid variable domain for dual constraint {variable}")

    raise NotImplementedError()


def _build_dual_objective(model: Model, dual_variables: list[Variable]) -> Objective:
    """
    Builds the dual objective from the primal model and the dual variables.

    :param model: primal model
    :type model: Model
    :param dual_variables: dual variables
    :type dual_variables: list[Variable]
    :return: dual objective
    :rtype: Objective
    """
    # TODO: build the dual objective from the primal model and the dual variables
    #
    # 1) create an empty expression for the dual objective
    # 2) for each primal constraint and the corresponding dual variable, add to the dual
    #    objective expression the product of the dual variable and the constraint bound
    # 3) create and return a new Objective with the dual objective expression and type MIN
    #
    # tip 1. you can use `zip()` to iterate over primal constraints and dual variables simultaneously

    dual_expression = Expression()
    for constraint, dual_var in zip(model.constraints, dual_variables):
        dual_expression += dual_var * constraint.bound

    return Objective(expression=dual_expression, type=ObjectiveType.MIN)
    #raise NotImplementedError()


def _build_dual_constraint(
    model: Model, variable: Variable, dual_variables: list[Variable]
) -> Constraint:
    # TODO: build a dual constraint for a primal variable from the primal model and the dual variables
    #
    # 1) create an empty expression for the dual constraint
    # 2) for each primal constraint and the corresponding dual variable
    #    a) extract the coefficient of the primal variable in the primal constraint expression
    #       using `constraint.expression.get_coefficient(variable)`
    #    b) add to the dual constraint expression the product of the coefficient and the dual variable
    # 3) extract the coefficient of the primal variable in the primal objective expression using `model.objective.expression.get_coefficient(variable)`
    # 4) create and return a new Constraint with the dual constraint expression, the objective coefficient as the bound,
    #    and the constraint type determined by `_dual_constraint_type(variable)`

    dual_expression = Expression()
    for constraint, dual_var in zip(model.constraints, dual_variables):
        coefficient = constraint.expression.get_coefficient(variable)
        dual_expression += dual_var * coefficient
    objective_coefficient = model.objective.expression.get_coefficient(variable)

    return Constraint(
        expression=dual_expression,
        type=_dual_constraint_type(variable),
        bound=objective_coefficient
    )

    #raise NotImplementedError()
