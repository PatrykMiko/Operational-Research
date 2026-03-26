import logging
from saport.linear_programming.model import Model
from saport.linear_programming.solvers.ortools.solver import OrToolsSolver
from saport.linear_programming.status import Status

def create_model() -> Model:
    model = Model("example_03_unbounded")

    # TODO:
    # fill missing test based on the example_01_solvable.py
    # to make the test a bit more interesting:
    # * make the model unbounded!
    x1 = model.create_variable("x1")
    x2 = model.create_variable("x2")

    model.add_constraint(x1 - x2 <= 10)

    model.minimize(-2 * x1 - 3 * x2)
    return model

def run():
    model = create_model()

    # TODO:
    # add a test "assert something" based on the example_01_solvable.py
    #
    # TIP: you may use other solvers (e.g. https://online-optimizer.appspot.com)
    #      to find the correct solution or use the OR-Tools solver available in the project
    # from saport.linear_programming.solvers.ortools.solver import OrToolsSolver
    solver = OrToolsSolver(model)

    try:
        result = solver.solve()
    except:
        raise AssertionError("This problem has a solution and your algorithm hasn't found it!")

    logging.info(result.solution)
    assert result.status == Status.UNBOUNDED, "The solution is unbounded!"

    logging.info("Congratulations! This solution seems to be alright :)")
    #raise AssertionError("Test is empty")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    run()
