from dataclasses import dataclass
from numpy.typing import NDArray
import numpy as np
from saport.linear_programming.model import Model

@dataclass
class Tableau:
    """
    A class representing a tableau for a simplex method.
    """

    model: Model
    """model corresponding to the tableaux"""
    table: NDArray[np.double]
    """2d-array containing the tableau"""

