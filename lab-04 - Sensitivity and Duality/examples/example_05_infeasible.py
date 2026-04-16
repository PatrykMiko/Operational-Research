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

    model.add_constraint(2*x1 - x2 <= -1)
    model.add_constraint(x1 + x2 == 3)
    model.add_constraint(x1 + x2 >= 4)
    
    model.maximize(x1 + 3 * x2)

    return model

def run():
    model = create_model()
    solver = Solver(model)

    try:
        result = solver.solve()
    except:
        raise AssertionError("Got an unexpected error while solving this problem. This shouldn't happen!")

    assert result.status == Status.INFEASIBLE, "Your algorithm has not detected an infeasible problem. This shouldn't happen..."
    logging.info("Congratulations! The solver seems to be working correctly :)")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    run()
