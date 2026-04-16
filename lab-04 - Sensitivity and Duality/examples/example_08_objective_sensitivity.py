# so one can run the example from the root of the project
import sys
sys.path.append('.')

import logging
import numpy as np
from saport.linear_programming.model import Model
from saport.linear_programming.solvers.simplex.solver import SimplexSolver as Solver
from saport.linear_programming.status import Status
from saport.linear_programming.transformations.duality import create_dual

def create_model() -> Model:
    model = Model(__file__)

    x1 = model.create_variable("x1")
    x2 = model.create_variable("x2")
    x3 = model.create_variable("x3")

    model.add_constraint(x1 + x2 + 2*x3 <= 4)
    model.add_constraint(2*x1 + x2 + 2*x3 <= 6)
    
    model.maximize(3*x1 + 2*x2 + 2.5*x3)
    
    return model

def run():
    model = create_model()
    solver = Solver(model)

    try:
        result = solver.solve()
    except:
        raise AssertionError("This problem has a solution and your algorithm hasn't found it!")

    assert result.status == Status.SOLVED, "The solution is infeasible!"
    assert result.solution is not None, "The solution is infeasible!"
    assert (result.solution.assignment == [2.0, 2.0, 0.0]), "Your algorithm found an incorrect solution!"

    cost_changes = result.solution.compute_cost_changes()
    assert len(cost_changes) == len(model.variables), "Cost changes should be computed for all original model variables!"
    assert cost_changes[0].variable.name == "x1", "Cost change for variable x1 is missing or has an incorrect variable name!"
    assert np.isclose(cost_changes[0].objective_delta_coefficient, 2.0), "Objective delta coefficient for variable x1 should be 2.0!"
    assert np.isclose(cost_changes[0].delta_min, -1.0), "The delta for variable x1 should not be less than `-1.0`!"
    assert np.isclose(cost_changes[0].delta_max, 1.0), "The delta for variable x1 should not be greater than `1.0`!"
    assert cost_changes[1].variable.name == "x2", "Cost change for variable x2 is missing or has an incorrect variable name!"
    assert np.isclose(cost_changes[1].objective_delta_coefficient, 2.0), "Objective delta coefficient for variable x2 should be 2.0!"
    assert np.isclose(cost_changes[1].delta_min, -0.5), "The delta for variable x2 should not be less than `-0.5`!"
    assert np.isclose(cost_changes[1].delta_max, 1.0), "The delta for variable x2 should not be greater than `1.0`!"
    assert cost_changes[2].variable.name == "x3", "Cost change for variable x3 is missing or has an incorrect variable name!"
    assert np.isclose(cost_changes[2].objective_delta_coefficient, 0.0), "Objective delta coefficient for variable x3 should be 0.0!"
    assert np.isclose(cost_changes[2].delta_min, float("-inf")), "The delta for variable x3 should not be bounded from below!"
    assert np.isclose(cost_changes[2].delta_max, 1.5), "The delta for variable x3 should not be bounded from above!"
    logging.info("Congratulations! This solution seems to be alright :)")
    logging.info("The cost sensitivity analysis for this model is:")
    logging.info("\n".join(str(change) for change in cost_changes))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    run()
