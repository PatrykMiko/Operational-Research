from dataclasses import dataclass
from enum import Enum
from typing import Optional


class DomainSign(Enum):
    """
    Classification of a variable domain by its sign restrictions.
    """

    NONPOSITIVE = "nonpositive"
    NONNEGATIVE = "nonnegative"
    FREE = "free"
    OTHER = "other"


@dataclass(frozen=True)
class Domain:
    """
    A class to represent the domain of a variable in linear programming.
    It can be used to specify the lower and upper bounds of a variable.
    """

    lower_bound: Optional[float]
    """the lower bound of the variable"""

    upper_bound: Optional[float]
    """the upper bound of the variable"""

    @property
    def is_normalized(self) -> bool:
        """
        Checks if the variable is normalized, i.e. if its lower bound is 0 and its upper bound is None.

        :return: True if the variable is normalized, False otherwise
        :rtype: bool
        """
        return self.lower_bound == 0 and self.upper_bound is None

    @property
    def sign(self) -> DomainSign:
        """
        Classifies the domain by its sign restrictions.

        :return: domain sign classification
        :rtype: DomainSign
        """
        if self.lower_bound is None and self.upper_bound == 0:
            return DomainSign.NONPOSITIVE
        if self.lower_bound == 0 and self.upper_bound is None:
            return DomainSign.NONNEGATIVE
        if self.lower_bound is None and self.upper_bound is None:
            return DomainSign.FREE
        return DomainSign.OTHER

    def __str__(self) -> str:
        left_part = f"[{self.lower_bound}" if self.lower_bound is not None else "(-inf"
        right_part = f"{self.upper_bound}]" if self.upper_bound is not None else "inf)"
        return f"{left_part}, {right_part}"
