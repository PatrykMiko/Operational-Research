import logging
from saport.linear_programming.model import Model
from saport.linear_programming.solvers.simplex.solver import SimplexSolver
from saport.linear_programming.status import Status


def run() -> None:
    model = Model("example_01_solvable")

    x1 = model.create_variable("x1")
    x2 = model.create_variable("x2")

    model.add_constraint(x1 <= 150)
    model.add_constraint(x2 <= 250)
    model.add_constraint(2*x1 + x2 <= 500)

    model.maximize(8 * x1 + 5 * x2)
    solver = SimplexSolver(model)

    try:
        result = solver.solve()
    except:
        raise AssertionError("This problem has a solution and your algorithm hasn't found it!")

    assert result.status == Status.SOLVED, "The solution is infeasible!"
    assert result.solution is not None, "The solution is infeasible!"
    assert (result.solution.assignment == [125.0, 250.0]), "Your algorithm found an incorrect solution!"

    logging.info("Congratulations! This solution seems to be alright :)")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    run()
