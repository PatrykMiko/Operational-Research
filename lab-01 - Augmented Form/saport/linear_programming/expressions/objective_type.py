from enum import Enum


class ObjectiveType(Enum):
    """
    An enum representing an objective type:
    """

    MAX = 1
    """maximize the objective"""
    MIN = -1
    """minimize the objective"""
    SAT = 0
    """just satisfy the constraints"""

    def __str__(self) -> str:
        return self.name.lower()
