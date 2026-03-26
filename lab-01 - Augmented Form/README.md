# Lab 01 - Linear Programming - Augmented Form

The goal of this lab is to:

- [ ] model all the assignments from the `assignment.pdf` in the corresponding functions of the `assignment.py` file
- [ ] fill missing code in the `saport.linear_programming.solvers.simplex.solver` module

## SAPORT

SAPORT = Student's Attempt to Produce an Operation Research Toolkit

Refer to the `example.py` for a quick overview of the API.

The project depends on numpy and Python (version >= `3.14` is required). All dependencies are listed in `pyproject.toml`. Use [uv](https://docs.astral.sh/uv/) to manage the Python version and dependencies automatically.

Short reminder:
- `uv run <command>` will run `<command>` within the correct environment;
- alternatively: `uv sync` will synchronize the environment and then `source .venv/bin/activate` will activate the python environment.

## Local Tests

To test your solutions off-line, you can:
1) use tips from `assignment.py` comments about optimal solutions and solve your models with external solver (example on-line solver is available [here](https://online-optimizer.appspot.com/?model=builtin:default.mod))
2) run the offline tests, i.e., just run the `uv run pytest` command in the project directory.

The off-line tests are designed just to help you develop your solutions. They are very basic and **are not** used in the grading process. The SAPORT uses the commercial [OR-Tools solver](https://developers.google.com/optimization) to check your models. You can check the code in `saport/linear_programming/solvers/ortools/solver.py` to see how our model is translated to an OR-Tools native model and how the OR-Tools solver is used.

## GitLab Setup

* [ ] Make sure, you have a **private** group
  * [how to create a group](https://docs.gitlab.com/ee/user/group/#create-a-group)
* [ ] Add @bobot-is-a-bot as the new group member (role: **maintainer**)
  * [how to add a group member](https://docs.gitlab.com/ee/user/group/#add-users-to-a-group)
* [ ] Fork this project into your new **private** group
  * [how to create a fork](https://docs.gitlab.com/ee/user/project/repository/forking_workflow.html#creating-a-fork)

## How To Submit Solutions

* [ ] Clone repository: git clone:
    ```bash
    git clone <repository url>
    ```
* [ ] Solve the exercises
    * remember to change only files with `#TODO` comments
* [ ] Commit your changes
    ```bash
    git add <path to the changed files>
    git commit -m <commit message>
    ```
* [ ] Push changes to the gitlab master branch
    ```bash
    git push 
    ```

The rest will be taken care of automatically. You can check the `GRADE.md` file for your grade / test results. Be aware that it may take some time till this file appears / gets updated. 

## Repository Guide

```bash
.
├── README.md        # this README
├── assignment.pdf   # pdf file with assignments to be modeled
├── assignment.py    # TODO: place to put the models of the assignments
├── conftest.py      # this file enables to call `pytest` without hustle
├── example.py       # just an example, how to create models in SAPORT
├── pyproject.toml   # package configuration and dependencies
├── saport           # directory with the SAPORT source-code
│   └── linear_programming  # directory containing code related to linear programming
│       ├── expressions        # directory with classes related to the LP model components
│       │   ├── atom.py        # atom is just a single variable with a coefficient
│       │   ├── constraint.py  # class representing linear constraint
│       │   ├── expression.py  # class representing linear expression
│       │   ├── objective.py   # class representing objective
│       │   └── variable.py    # class representing a decision variable
│       ├── exceptions.py # project specific exceptions
│       ├── model.py      # class allowing to create linear programming models
│       ├── result.py     # class wrapping solver result with status
│       ├── solution.py   # class representing solutions to the problems
│       ├── solver.py     # abstract base class for solvers
│       ├── status.py     # enum representing solver status
│       └── solvers       # directory with concrete solver implementations
│           ├── ortools/solver.py   # OR-Tools based solver
│           └── simplex             # simplex algorithm implementation
│               ├── constants.py    # solver constants
│               ├── simplex_solution.py # extended solution with tableau info
│               ├── solver.py     # TODO: simplex solver, you have to fill methods here
│               └── tableau.py    # class representing the simplex tableau
├── test_screenshot.png # just a screenshot used in the README
└── tests               # directory with local tests
```
