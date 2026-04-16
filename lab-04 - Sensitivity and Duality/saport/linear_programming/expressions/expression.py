from typing import Sequence, TYPE_CHECKING

from itertools import groupby
from functools import reduce


import saport.linear_programming.expressions.constraint as ConstraintM
from saport.linear_programming.expressions.constraint_type import ConstraintType
import saport.linear_programming.expressions.atom as AtomM

if TYPE_CHECKING:
    from saport.linear_programming.expressions.affine_substitution import (
        AffineSubstitution,
    )
    from saport.linear_programming.expressions.variable import Variable


class Expression:
    """
    A class to represent a linear polynomial, i.e. a sum of atom (e.g. 4x + 5y - 0.4z)
    """

    atoms: list[AtomM.Atom]
    """list of atoms in the polynomial (atom is a single variable with a coefficient)"""
    _coefficients: dict[int, float]
    """ a private cache mapping variables' indices to their coefficients — do not use directly!"""

    def __init__(self, *atoms: AtomM.Atom):
        """
        Initializes the expression with one or more atoms.

        :param atoms: one or more of atoms
        :type atoms: Atom
        """
        self.atoms = list(atoms)
        coefficients: dict[int, float] = dict()
        for atom in self.atoms:
            coefficient = coefficients.get(atom.var.index, 0.0)
            coefficients[atom.var.index] = coefficient + atom.coefficient
        self._coefficients = coefficients

    @classmethod
    def from_vectors(
        cls, variables: Sequence[Variable], coefficients: Sequence[float]
    ) -> Expression:
        """
        Constructs an expression from collections of coefficients and corresponding variables.

        :param variables: a collection of variables
        :type variables: Sequence[Variable]
        :param coefficients: a collection of coefficient of the same length as the `variables` parameter
        :type coefficients: Sequence[float]
        :return: a new expression
        :rtype: Expression
        """
        assert len(variables) == len(coefficients), (
            "number of coefficients should correspond to variables in the expression"
        )
        atoms = [AtomM.Atom(v, f) for (v, f) in zip(variables, coefficients)]
        return Expression(*atoms)

    def evaluate(self, assignment: Sequence[float]) -> float:
        """
        Returns value of the expression for the given assignment
        Assignment is just a list of values with order corresponding to the variables in the model,
        e.g. 2x+4y+5z has value for assignment (1,2,3) = 2 * 1 + 4 * 2 + 5 * 3 = 25, assuming
             x has value 1, y has value 2, z has value 3

        :param assignment: a collection containing values of the variables
        :type assignment: Sequence[float]
        :return: a value of the expression
        :rtype: float
        """

        def adder(val: float, a: AtomM.Atom) -> float:
            return val + a.evaluate_with_value(assignment[a.var.index])

        return reduce(adder, self.atoms, 0.0)

    def simplify(self):
        """
        It's an in-place operation: it sorts atoms and reduces coefficients, so there
        are no two atoms involving the same variable.
        """

        def projection(a: AtomM.Atom) -> int:
            return a.var.index

        def reduce_atoms(a1: AtomM.Atom, a2: AtomM.Atom) -> AtomM.Atom:
            return AtomM.Atom(a1.var, a1.coefficient + a2.coefficient)

        def reduce_group(g: Sequence[AtomM.Atom]) -> AtomM.Atom:
            return reduce(reduce_atoms, g[1:], g[0])

        sorted_atoms = sorted(self.atoms, key=projection)
        grouped_atoms = [list(g[1]) for g in groupby(sorted_atoms, key=projection)]
        self.atoms = [reduce_group(g) for g in grouped_atoms]

    def get_coefficients(self, variables: Sequence[Variable]) -> list[float]:
        """
        Converts an expression to a flat list of the coefficients, e.g.
        - given:
            variables, x1, x2, x3, x4
            expression 2x1 + 3x4
        - result should be
            [2.0, 0.0, 0.0, 4.0]
            as variables:
                * x1 has coefficient 2.0
                * x2 and x3 are missing from expression, so their coefficients are zero
                * x4 has coefficient 4.0

        :param variables: a collection of variables
        :type variables: list[Variable]
        :return: list of the coefficients returned in the same order as the input variables
        :rtype: list[float]
        """
        return [self.get_coefficient(v) for v in variables]

    def get_coefficient(self, var: Variable) -> float:
        """
        Returns a coefficient of the given variable in the expression

        :param var: a single variable
        :type var: Variable
        :return: a coefficient of the variable
        :rtype: float
        """
        return self._coefficients.get(var.index, 0.0)

    def set_coefficient(self, var: Variable, coefficient: float):
        """
        Replaces in-place a coefficient of the given variable in the equation.
        overrides coefficient for the given variable.
        If there is no such variable in the expression, it's get added with the given coefficient.
        Setting coefficient to 0.0 removes variable from the expression.
        As a side effect all atoms with the variable `var` get simplified.

        :param var: variable, we want to change the coefficient of
        :type var: Variable
        :param coefficient: a new coefficient value
        :type coefficient: float
        """
        self.atoms = [a for a in self.atoms if a.var != var]
        self._coefficients[var.index] = coefficient
        if coefficient != 0.0:
            self.atoms.append(AtomM.Atom(var, coefficient))

    def substitute(
        self, substitutions: dict[Variable, AffineSubstitution]
    ) -> tuple[Expression, float]:
        """
        Returns a new expression with variables replaced by affine substitutions.

        :param substitutions: substitutions to apply
        :type substitutions: dict[Variable, AffineSubstitution]
        :return: a pair consisting of the substituted expression and the introduced constant shift
        :rtype: tuple[Expression, float]
        """
        new_atoms: list[AtomM.Atom] = []
        constant_shift = 0.0

        for atom in self.atoms:
            substitution = substitutions.get(atom.var)
            if substitution is None:
                new_atoms.append(atom)
                continue

            new_atoms += (substitution.expression * atom.coefficient).atoms
            constant_shift += atom.coefficient * substitution.offset

        return (Expression(*new_atoms), constant_shift)

    def __add__(self, other: Expression | float) -> Expression:
        """
        Returns a sum of the two polynomials.
        It also accepts adding an empty Expression, e.g., a `0`.

        :param other: a polynomial we want to add or a zero
        :type other: Expression | float
        :return: a new expression
        :rtype: Expression
        """
        if isinstance(other, (int, float)) and other == 0:
            return Expression(*self.atoms)
        assert isinstance(other, Expression), (
            f"expressions supports only adding 0 as a numeric value, got {other}"
        )
        new_atoms = list(self.atoms)
        new_atoms += other.atoms
        return Expression(*new_atoms)

    __radd__ = __add__

    def __sub__(self, other: Expression) -> Expression:
        """
        Returns a sum of the two polynomials, inverting the second polynomial
        Useful for expressions like 3*x - 4y, otherwise one would have to write 3*x + -4*y

        :param other: a polynomial we want to subtract
        :type other: Expression
        :return: a new expression
        :rtype: Expression
        """
        return self.__add__(other * -1)

    def __neg__(self) -> Expression:
        """
        Negates the polynomial, creating a new one.
        Effective acts as multiplying by `-1`

        :return: a new expression
        :rtype: Expression
        """
        return self.__mul__(-1)

    def __mul__(self, factor: float) -> Expression:
        """
        Returns a new polynomial with all coefficients multiplied by the given number

        :param factor: a number the expression is to be multiplied by
        :type factor: float
        :return: a new expression
        :rtype: Expression
        """
        new_atoms = [a * factor for a in self.atoms]
        return Expression(*new_atoms)

    __rmul__ = __mul__

    def __eq__(self, bound: float) -> ConstraintM.Constraint:  # type: ignore[override]
        """
        Creates a new "equal to" constraint with the expression
        being a left side of equation.

        :param bound: a right side of a "==" equation
        :type bound: float
        :return: a new constraint
        :rtype: Constraint
        """
        return ConstraintM.Constraint(self, bound, ConstraintType.EQ)

    def __ge__(self, bound: float) -> ConstraintM.Constraint:
        """
        Creates a new "greater than or equal" constraint with the expression
        being a left side of equation.

        :param bound: a right side of a ">=" equation
        :type bound: float
        :return: a new constraint
        :rtype: Constraint
        """
        return ConstraintM.Constraint(self, bound, ConstraintType.GE)  # type: ignore

    def __le__(self, bound: float) -> ConstraintM.Constraint:
        """
        Creates a new "less than or equal" constraint with the expression
        being a left side of equation.

        :param bound: a right side of a "<=" equation
        :type bound: float
        :return: a new constraint
        :rtype: Constraint
        """
        return ConstraintM.Constraint(self, bound, ConstraintType.LE)  # type: ignore

    def __str__(self) -> str:
        text = str(self.atoms[0])

        for atom in self.atoms[1:]:
            text += " + " if atom.coefficient >= 0 else " - "
            coefficient = (
                "" if abs(atom.coefficient) == 1.0 else f"{abs(atom.coefficient)}*"
            )
            text += f"{coefficient}{atom.var.name}"
        return text
