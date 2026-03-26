from abc import ABC, abstractmethod
from dataclasses import dataclass
from saport.linear_programming.result import Result
from saport.linear_programming.model import Model
from saport.linear_programming.solution import Solution


@dataclass
class Solver[S: Solution](ABC):
    """
    An abstract class to be inherited from by any linear programming solver.
    It is generic over its solution type as various solvers may return less or more detailed solutions.
    """

    model: Model
    """A linear programming model to be solved"""

    @abstractmethod
    def solve(self) -> Result[S]:
        """
        Solves a given linear programming model.

        :param model: a linear programming model
        :type model: Model
        :return: a result containing status and (hopefully) a solution
        :rtype: Result[S]
        """
