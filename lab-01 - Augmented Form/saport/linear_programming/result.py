from saport.linear_programming.solution import Solution
from saport.linear_programming.status import Status


from dataclasses import dataclass


@dataclass
class Result[S: Solution]:
    """
    Represents a result of the solver.
    """

    status: Status
    """what is the solver's status"""
    solution: S | None
    """what solution has been found, `None` if the status is not `SOLVED`"""

    @classmethod
    def with_solution(cls, solution: S) -> Result[S]:
        """
        A helper constructor to build a result with a valid solution

        :param solution: a solution to the model
        :type solution: Solution
        :return: a result object with a status.SOLVED and corresponding solution
        :rtype: Result
        """
        return Result(Status.SOLVED, solution)

    @classmethod
    def unbounded(cls) -> Result[S]:
        """
        A helper constructor to build a result for an unbounded model

        :return: a result object with a status.UNBOUNDED and no solution
        :rtype: Result
        """
        return Result(Status.UNBOUNDED, None)

    @classmethod
    def infeasible(cls) -> Result[S]:
        """
        A helper constructor to build a result for an infeasible model

        :return: a result object with a status.INFEASIBLE and no solution
        :rtype: Result
        """
        return Result(Status.INFEASIBLE, None)

    @classmethod
    def errored(cls) -> Result[S]:
        """
        A helper constructor to build a result for the error situations

        :return: a result object with a status.ERROR and no solution
        :rtype: Result
        """
        return Result(Status.ERROR, None)
