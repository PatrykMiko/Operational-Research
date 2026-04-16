# so one can run the example from the root of the project
import sys
sys.path.append('.')

import logging
from saport.linear_programming.model import Model
from saport.linear_programming.solvers.simplex.solver import SimplexSolver as Solver
from saport.linear_programming.status import Status


def create_model() -> Model:
    model = Model(__file__)
    
    x1 = model.create_variable("x1")
    x2 = model.create_variable("x2")

    model.add_constraint(5*x1 + 3*x2 <= 30)
    model.add_constraint(4*x1 + 2*x2 <= 18)
    model.maximize(10*x1 + 6*x2)
    
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
    assert (result.solution.assignment == [0.0, 9.0]), "Your algorithm found an incorrect solution!"
    assert not result.solution.is_unique(), "This problem has multiple solutions, but your algorithm claims it is unique!"
    logging.info("Congratulations! This solution seems to be alright :)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    run()
