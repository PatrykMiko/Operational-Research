# so one can run the example from the root of the project
import sys
sys.path.append('.')

import logging
from saport.linear_programming.model import Model
from saport.linear_programming.solvers.simplex.solver import SimplexSolver as Solver
from saport.linear_programming.status import Status

def create_model() -> Model:
    model = Model(__file__)
    
    # you can specify the lower and upper bounds of the variables when creating them.
    # if you don't specify them, they will be set to 0 and infinity respectively by default.
    x1 = model.create_variable("x1", lb=5, ub=10)

    # here we set the lower bound to negative infinity and the upper bound to 8.
    x2 = model.create_variable("x2", lb=None, ub=8)

    # this is a default behavior made explicit for demonstration purposes.
    # it means same as x3 = model.create_variable("x3") or
    x3 = model.create_variable("x3", lb=0, ub=None)

    model.add_constraint(x1 + x2 >= 3)
    model.add_constraint(x1 + x3 <= 15)
    
    model.maximize(x1 - 3 * x2 + 2 * x3)
    
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
    assert (result.solution.assignment == [10.0, -7.0, 5.0]), "Your algorithm found an incorrect solution!"

    logging.info("Congratulations! This solution seems to be alright :)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    run()
