# Lab 04 - Duality and Sensitivity Analysis

The goal of this lab is to implement algorithms for creating a dual model and performing the cost-change sensitivity analysis.
To achieve this, you will need to fill missing code in the following files:

* `saport.linear_programming.solvers.simplex.tableau`,
* `saport.linear_programming.solvers.simplex.simplex_solution`,
* `saport.linear_programming.transformations.duality`.


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
├── README.md                # this README
├── conftest.py              # this file enables to call `pytest` without hustle
├── example.py               # just an example, how to create models in SAPORT
├── pyproject.toml           # package configuration and dependencies
└── saport                   # directory with the SAPORT source-code
    ├── exceptions.py        # project specific exceptions
    ├── expressions          # directory with classes related to the LP model components
    │   ├── affine_substitution.py # class for variable substitution/transformations
    │   ├── atom.py          # atom is just a single variable with a coefficient
    │   ├── constraint.py    # class representing a linear constraint
    │   ├── constraint_type.py # enum for constraint types (e.g., <=, >=, ==)
    │   ├── domain.py        # class representing the domain bounds of a variable
    │   ├── expression.py    # class representing a linear expression
    │   ├── objective.py     # class representing an objective function
    │   ├── objective_type.py # enum for objective types (minimize/maximize/satisfy)
    │   └── variable.py      # class representing a decision variable
    ├── model.py             # class allowing to create linear programming models
    ├── result.py            # class wrapping solver result with status
    ├── solution.py          # class representing solutions to the problems
    ├── solver.py            # abstract base class for solvers
    ├── solvers              # directory with concrete solver implementations
    │   ├── ortools
    │   │   └── solver.py    # OR-Tools based solver
    │   └── simplex          # simplex algorithm implementation
    │       ├── constants.py # solver constants
    │       ├── cost_change.py # cost-change sensitivity analysis result for a single variable
    │       ├── simplex_solution.py # TODO: simplex solution class, you have to fill methods here
    │       ├── solver.py    # simplex solver implementation
    │       └── tableau.py   # TODO: class representing the simplex tableau, you have to fill methods here
    ├── status.py            # enum representing solver status
    └── transformations      # directory for altering and converting LP models
        ├── duality.py       # TODO: creating the dual model, you have to fill methods here
        └── variable_preprocessing.py # variable ranges preprocessing logic
```
