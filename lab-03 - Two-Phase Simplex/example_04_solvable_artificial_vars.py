import logging
from saport.linear_programming.model import Model
from saport.linear_programming.solvers.simplex.solver import SimplexSolver as Solver
from saport.linear_programming.status import Status
from saport.linear_programming.solvers.simplex.solver import SimplexSolver as Solver

def create_model() -> Model:
    model = Model("example_04_solvable_artificial")

    # TODO:
    # fill missing test based on the example_02_solvable.py
    # to make the test a bit more interesting:
    # * make sure solver has to use some artificial variables

    x1 = model.create_variable("x1")
    x2 = model.create_variable("x2")
    x3 = model.create_variable("x3")

    model.add_constraint(x1 + 3 * x2 + 2 * x3 <= 10)
    model.add_constraint(-1 * x1 - 5 * x2 - 1 * x3 <= -8)

    model.minimize(-8 * x1 - 10 * x2 - 7 * x3)
    
    return model

def run():
    model = create_model()

    # TODO:
    # add a test "assert something" based on the example_02_solvable.py
    # TIP: you may use other solvers (e.g. https://online-optimizer.appspot.com)
    #      to find the correct solution or use the OR-Tools solver available in the project
    # from saport.linear_programming.solvers.ortools.solver import OrToolsSolver

    solver = Solver(model)

    try:
        result = solver.solve()
    except:
        raise AssertionError("This problem has a solution and your algorithm hasn't found it!")

    assert result.status == Status.SOLVED, "The solution is infeasible!"
    assert result.solution is not None, "The solution is infeasible!"
    assert (result.solution.assignment == [10.0, 0.0, 0.0]), "Your algorithm found an incorrect solution!"

    logging.info("Congratulations! This solution seems to be alright :)")
    #logging.info("This test is empty but it shouldn't be, fix it!")
    #raise AssertionError("Test is empty")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    run()
