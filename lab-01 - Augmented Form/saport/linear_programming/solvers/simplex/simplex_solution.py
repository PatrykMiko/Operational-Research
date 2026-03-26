from saport.linear_programming.solution import Solution
from saport.linear_programming.solvers.simplex.tableau import Tableau


from dataclasses import dataclass


@dataclass
class SimplexSolution(Solution):
    """
    A more detailed variant of the basic solution class.
    Contains information about the tableau used by a simplex solver.
    """

    initial_tableau: Tableau
    """
    Contains an initial tableau for the solved model.
    Useful for debugging.
    """
    tableau: Tableau
    """
    Contains the final tableau for the solved model.
    """
