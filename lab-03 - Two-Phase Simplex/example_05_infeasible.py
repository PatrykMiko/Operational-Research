import logging
from saport.linear_programming.model import Model
from saport.linear_programming.solvers.simplex.solver import SimplexSolver as Solver
from saport.linear_programming.status import Status

def create_model() -> Model:
    model = Model("example_05_infeasible")

    # TODO:
    # fill missing test based on the example_03_unbounded.py
    # to make the test a bit more interesting:
    # * make sure model is infeasible

    x1 = model.create_variable("x1")
    x2 = model.create_variable("x2")
    x3 = model.create_variable("x3")

    model.add_constraint(x1 + x2 + x3 <= 10)
    model.add_constraint(x1 + x2 + x3 >= 11)

    model.maximize(5 * x1 + 8 * x2 + 7 * x3)

    return model

def run():
    model = create_model()
    # TODO:
    # add a test "assert something" based on the example_03_unbounded.py
    # TIP: you may use other solvers (e.g. https://online-optimizer.appspot.com)
    #      to find the correct solution or use the OR-Tools solver available in the project
    # from saport.linear_programming.solvers.ortools.solver import OrToolsSolver

    solver = Solver(model)

    try:
        result = solver.solve()
    except:
        raise AssertionError("Got an unexpected error while solving this problem. This shouldn't happen!")

    assert result.status == Status.SOLVED, "Your algorithm has not detected an infeasible problem. This shouldn't happen..."
    assert result.solution is not None, "Your algorithm has not detected an infeasible problem. This shouldn't happen..."
    logging.info("Congratulations! The solver seems to be working correctly :)")

    #logging.info("This test is empty but it shouldn't be, fix it!")
    #raise AssertionError("Test is empty")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    run()
