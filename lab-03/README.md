# Lab 03 - Two-Phase Simplex

The goal of this lab is to implement a simplex solver that handles all linear problems. To achieve that, one has to:

* fill missing code in the `saport.linear_programming.solvers.simplex.solver.SimplexSolver` class
* create two models to test the algorithm, filling missing code in the `example_04_solvable_artificial_vars.py` and `example_05_infeasible.py` files

## SAPORT

SAPORT = Student's Attempt to Produce an Operation Research Toolkit

Refer to the `example.py` for a quick overview of the API.

The project depends on numpy and Python (version >= `3.14` is required). All dependencies are listed in `pyproject.toml`. Use [uv](https://docs.astral.sh/uv/) to manage the Python version and dependencies automatically.

Short reminder:
- `uv run <command>` will run `<command>` within the correct environment;
- alternatively: `uv sync` will synchronize the environment and then `source .venv/bin/activate` will activate the python environment.

## How To Run Local Tests

### Solver Class

Run the tests with `uv run pytest` command in the project directory.

### Examples

The example files are very basic acceptance tests and **are not** used in the grading process.
*Warning*: to run the examples, you should first implement the simplex algorithm or use the OR-Tools solver. Otherwise the examples will fail due to unimplemented methods in the `SimplexSolver`.

Then you can just run every example using `uv run python path_to_example.py`, or you can run all the examples with: `uv run python test.py`.

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
├── conftest.py      # this file enables to call `pytest` without hustle
├── example.py       # just an example, how to create models in SAPORT
├── example_01_solvable.py  # an example of a naive acceptance test for our solver
├── example_02_solvable.py  # same as ^
├── example_03_unbounded.py # same as ^
├── example_04_solvable_artificial_vars.py  # TODO: fill this example based on the example_02...
├── example_05_infeasible.py # TODO: fill this example based on the example_03...
├── test.py          # runs all the examples
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
└── tests               # directory with local tests
```
