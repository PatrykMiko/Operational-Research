from enum import Enum, auto


class Status(Enum):
    """
    Represents possible solving statuses after running the solver.
    """

    SOLVED = auto()
    """the model has been solved successfully"""
    INFEASIBLE = auto()
    """the model is infeasible"""
    UNBOUNDED = auto()
    """the model is unbounded"""
    ERROR = auto()
    """an unexpected error has occurred"""
