import logging
from saport.linear_programming.model import Model
from saport.linear_programming.solvers.simplex.solver import SimplexSolver as Solver
from saport.linear_programming.status import Status


def create_model() -> Model:
    model = Model("example_03_unbounded")

    x1 = model.create_variable("x1")
    x2 = model.create_variable("x2")
    x3 = model.create_variable("x3")

    model.add_constraint(x1 - 3*x2 - 2*x3 <= 10)
    model.add_constraint(x1 + 5*x2 - 1*x3 <= 7)

    model.maximize(5 * x1 + 8 * x2 + 7 * x3)

    return model

def run():
    model = create_model()
    solver = Solver(model)

    try:
        result = solver.solve()
    except:
        raise AssertionError("Got an unexpected error while solving this problem. This shouldn't happen!")

    assert result.status == Status.UNBOUNDED, "Your algorithm has not detected an unbounded problem. This shouldn't happen..."
    logging.info("Congratulations! The solver seems to be working correctly :)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    run()
