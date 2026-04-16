from enum import Enum


class ConstraintType(Enum):
    """
    An enum to represent a constraint type
    """

    LE = -1
    """less than or equal"""
    EQ = 0
    """equal to"""
    GE = 1
    """greater than or equal"""

    def __str__(self):
        return {
            ConstraintType.LE: "<=",
            ConstraintType.EQ: "==",
            ConstraintType.GE: ">=",
        }[self]
