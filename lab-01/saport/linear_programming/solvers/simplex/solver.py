from copy import deepcopy
import numpy as np

from saport.linear_programming.result import Result
from saport.linear_programming.expressions import (
    Variable,
    ObjectiveType,
    ConstraintType,
    Constraint,
)
from saport.linear_programming.model import Model
from saport.linear_programming.solver import Solver as LinearProgrammingSolver
from saport.linear_programming.solvers.simplex.constants import eps
from saport.linear_programming.solvers.simplex.simplex_solution import SimplexSolution
from saport.linear_programming.solvers.simplex.tableau import Tableau


class SimplexSolver(LinearProgrammingSolver[SimplexSolution]):
    """
    A class to represent a simplex solver.
    """

    _slacks: dict[Variable, Constraint]
    """contains mapping from slack variables to their corresponding constraints"""
    _surpluses: dict[Variable, Constraint]
    """contains mapping from surplus variables to their corresponding constraints"""
    _artificial: dict[Variable, Constraint]
    """contains mapping from artificial variables to their corresponding constraints"""

    def solve(self) -> Result[SimplexSolution]:
        """
        Solves the given linear model.

        :param model: a linear programming model to be solved
        :type model: Model
        :return: an object representing the solution
        :rtype: Solution
        """
        self.model.validate()
        normal_model = self._augment_model(self.model)
        tableau = self._basic_initial_tableau(normal_model)
        raise NotImplementedError("will be expanded in the next lab")



    def _augment_model(self, original_model: Model) -> Model:
        """
        Creates an augmented model, satisfying the simplex method requirements.

        :param original_model: a linear model as defined by the user
        :type original_model: Model
        :return: a simplex-friendly model (max objective, == constraints, nonnegative bounds)
        :rtype: Model
        """
        # We don't want to modify the original model, so we copy it
        model = deepcopy(original_model)
        # Wa want to have simplified expressions
        # each variable should occur only once in every expression
        model.simplify()

        # Let's make sure the model wants to maximize the objective
        self._change_objective_to_max(model)
        self._change_constraints_bounds_to_nonnegative(model)
        self._add_slack_and_surplus_variables(model)
        return model
    

    def _change_objective_to_max(self, model: Model) -> None:
        """
        Modifies a given model in-place to have a "maximize" objective

        :param model: the model to be modified
        :type model: Model
        """
        # TODO:
        # if the user wants to minimize, we convert: min f(x) ≡ max -f(x).
        #
        # - if the objective is minimizing, we have to "invert" it
        #   tip 0. Objective is stored in model.objective
        #   tip 1. Objective has `type` attribute storing ObjectiveType (.MIN / .MAX)
        #   tip 2. Objective class has an `invert` method just for this purpose
        if model.objective.type == ObjectiveType.MIN:
            model.objective.invert()

    def _change_constraints_bounds_to_nonnegative(self, model: Model):
        """
        Modifies a given model in-place to have only non-negative bounds.

        :param model: a model to be modified
        :type model: Model
        """
        # TODO:
        # all the bounds in the augmented model have to be positive or equal zero
        # - every constraint with a negative bound has to be "inverted"
        #   tip 0. Constraints are stored in model.constraints
        #   tip 1. Constraint class has an "invert" method just for this purpose
        for constraint in model.constraints:
            if constraint.bound < 0:
                constraint.invert()

    def _add_slack_and_surplus_variables(self, model: Model) -> None:
        """
        Augments a given model with slack and surplus variables.
        The new variables are then stored in the self._slacks and self._surpluses dictionaries,
        mapping them to their corresponding constraints.

        :param model: a model to be augmented
        :type model: Model
        """
        # empty dictionaries to store informations about slacks / surplus variables
        slacks: dict[Variable, Constraint] = dict()
        surpluses: dict[Variable, Constraint] = dict()

        # TODO:
        # the simplex method operates on equations, not inequalities. We convert
        # each inequality into an equation by adding a new "helper" variable:
        # - slack variables absorb the leftover in `<=` constraints (e.g. x1 + x2 <= 10  →  x1 + x2 + s = 10)
        # - surplus variables absorb the excess in `>=` constraints (e.g. x1 + x2 >= 5  →  x1 + x2 - s = 5)
        #
        # iterate through the constraints and:
        # - if constraint has type ConstraintType.EQ, we can ignore it
        # - if constraint has type ConstraintType.LE:
        #   * create a new slack variable in the model
        #   * add this variable to the constraint's expression
        #   * store variable in the `slacks` dictionary - the new variable is the key, the constraint is the value
        #   * change type of constraint to ConstraintType.EQ
        # - if constraint has type ConstraintType.GE:
        #   * create a new surplus variable in the model
        #   * subtract this variable from the constraint's expression
        #   * store variable in the `surpluses` dictionary - the new variable is the key, the constraint is the value
        #   * change type of constraint to ConstraintType.EQ
        #
        # tip 0. constraints are stored in model.constraints
        # tip 1. constraint has properties `type`, `expression`, and unique `index`
        # tip 2. '-' and '+' operators are overloaded for the expression type, so you can literally add/subtract variables
        #        to the expression. You can also use in-place versions: "-=" and "+="
        # tip 3. every variable in model should have a unique name, for now let's assume names `s1`, `s2`, `s3`, ..., are fine
        #        (use model.create_variable to create them)
        s_index = 1
        for constraint in model.constraints:
            if constraint.type == ConstraintType.LE:
                slack_val = model.create_variable(f"s{s_index}")
                s_index += 1
                constraint.expression += slack_val
                slacks[slack_val] = constraint
                constraint.type = ConstraintType.EQ
            elif constraint.type == ConstraintType.GE:
                surplus_val = model.create_variable(f"s{s_index}")
                s_index += 1
                constraint.expression -= surplus_val
                surpluses[surplus_val] = constraint
                constraint.type = ConstraintType.EQ

        # we remember info about the slack and surpluses, it will be useful in the next class
        self._slacks = slacks
        self._surpluses = surpluses


    def _basic_initial_tableau(self, model: Model) -> Tableau:
        """
        Creates a trivial initial tableau for the models satisfying some basic requirements.

        :param model: a model having a trivial initial solution
        :type model: Model
        :return: an trivial initial tableau
        :rtype: Tableau
        """
        # TODO:
        # the simplex tableau is the matrix representation of our linear program.
        # the first row encodes the objective, and each subsequent row encodes a constraint.
        #
        # Replace the 'None' below with a numpy array, where:
        # 1) first row consists of the inverted coefficients of the objective expression
        #    plus 0.0 in the last column
        # 2) every other row consists of the coefficients in the corresponding constraints,
        #    don't forget to put the constraint bound in the last column
        #
        # tips:
        # - to invert coefficients in an expression, one can multiply it by `-1`
        # - to get coefficients one can use the `expression.get_coefficients` method on `model.variables`,
        #   it will return a list of floats
        #
        # - one can easily create an array based on an existing list, e.g.
        #   * having list of lists: a = [[1,2,3], [4,5,6], [7,8,9]]
        #   * using `np.array(a)` will result in corresponding array:
        #       |1 2 3|
        #       |4 5 6|
        #       |7 8 9|

        obj_coeffs = model.objective.expression.get_coefficients(model.variables)
        objective_row = [-1 * c for c in obj_coeffs] + [0.0]
        matrix_data = [objective_row]

        for constraint in model.constraints:
            constraint_row = constraint.expression.get_coefficients(model.variables) + [constraint.bound]
            matrix_data.append(constraint_row)

        table = np.array(matrix_data)
        return Tableau(model, table)

