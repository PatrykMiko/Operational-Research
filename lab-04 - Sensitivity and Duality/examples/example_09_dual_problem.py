# so one can run the example from the root of the project
import sys
sys.path.append('.')

import logging
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

    # create and solve the dual problem
    dual_model = create_dual(model)
    dual_solver = Solver(dual_model)
    dual_result = dual_solver.solve()

    assert dual_result.status == Status.SOLVED, "The dual problem is infeasible!"
    assert dual_result.solution is not None, "The dual problem is infeasible!"
    assert (dual_result.solution.assignment == [1.0, 1.0]), "Your algorithm found an incorrect solution for the dual problem!"      
    assert dual_result.solution.objective_value() == result.solution.objective_value(), "The optimal values of the primal and dual problems should be equal!"


    logging.info("Congratulations! This solution seems to be alright :)")
    logging.info("The dual problem for this model is:")
    logging.info(dual_model)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    run()
