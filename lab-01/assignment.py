from saport.linear_programming.model import Model
from saport.linear_programming.solvers.ortools.solver import OrToolsSolver

# TODO:
# Model assignments from assignment.pdf
# tip 1. you may use the or-tools solver to check if your models reach correct optima
# tip 2. you can also use an external solver available on-line: https://online-optimizer.appspot.com/?model=builtin:default.mod


def assignment_1():
    model = Model("Assignment 1")

    # TODO:
    # Add:
    # - variables
    x1 = model.create_variable("x1")
    x2 = model.create_variable("x2")
    x3 = model.create_variable("x3")
    # - constraints
    model.add_constraint(x1 + x2 + x3 <= 30)
    model.add_constraint(x1 + 2*x2 + x3 >= 10)
    model.add_constraint(2*x2 + x3 <= 20)
    # - objective
    model.maximize(2 * x1 + x2 + 3 * x3)
    # tip. value at optimum: 80.0
    return model


def assignment_2():
    model = Model("Assignment 2")

    # TODO:
    # Add:
    # - variables
    p1 = model.create_variable("p1")
    p2 = model.create_variable("p2")
    p3 = model.create_variable("p3")
    p4 = model.create_variable("p4")
    # - constraints
    model.add_constraint(0.8*p1 + 2.4*p2 + 0.9*p3 + 0.4*p4 >= 1200)
    model.add_constraint(0.6*p1 + 0.6*p2 + 0.3*p3 + 0.3*p4 >= 600)
    # - objective
    model.minimize(9.6 * p1 + 14.4 * p2 + 10.8* p3 + 7.2*p4)
    # tip. value at optimum: 10800.0
    return model


def assignment_3():
    model = Model("Assignment 3")

    # TODO:
    # Add:
    # - variables
    x1 = model.create_variable("x1")
    x2 = model.create_variable("x2")
    # - constraints
    model.add_constraint(5*x1 + 15*x2 >= 50)
    model.add_constraint(20*x1 + 5*x2 >= 40)
    model.add_constraint(15*x1 + 2*x2 <= 60)
    # - objective
    model.minimize(8*x1 + 4*x2)
    # tip. value at optimum: 21.8181818
    return model


def assignment_4():
    model = Model("Assignment 4")

    # TODO:
    # Add:
    # - variables
    x12 = model.create_variable("x12")
    x133 = model.create_variable("x133")
    x223 = model.create_variable("x223")
    x2333 = model.create_variable("x2333")
    x33333 = model.create_variable("x33333")
    # - constraints
    model.add_constraint(x12+x133>=150)
    model.add_constraint(x12+2*x223+x2333>=200)
    model.add_constraint(2*x133+x223+3*x2333+5*x33333>=150)
    # - objective
    model.minimize(20*x12+25*x133+15*x223+20*x2333+25*x33333)
    # tip. value at optimum: 4000.0
    return model
