from dataclasses import dataclass

from saport.linear_programming.expressions.variable import Variable


@dataclass
class CostChange:
    """
    Represents the effect of changing a tableau variable cost coefficient.
    """

    variable: Variable
    """the tableau variable whose cost coefficient is changed"""

    objective_delta_coefficient: float
    """
    the multiplier describing how the objective value changes with delta:
    new_objective = old_objective + objective_delta_coefficient * delta
    """

    delta_min: float
    """minimum delta that preserves the current optimal basis"""

    delta_max: float
    """maximum delta that preserves the current optimal basis"""

    def __str__(self) -> str:
        return (
            f"Cost change for variable {self.variable.name}: "
            f"objective_delta_coefficient={self.objective_delta_coefficient:.3f}, "
            f"delta_min={self.delta_min:.3f}, delta_max={self.delta_max:.3f}"
        )
