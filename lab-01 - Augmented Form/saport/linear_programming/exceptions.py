class CalledWithoutAssignmentError(Exception):
    """
    A method requiring an value assignment present has been called without it.
    """

    pass


class DuplicateVariableError(Exception):
    """
    One has tried to create two variables with the same name inside a single model.
    """

    def __init__(self, name: str) -> None:
        super().__init__(
            f"Cannot create variable named {name}. There is already a variable with the same name."
        )
        self.name = name


class EmptyModelError(Exception):
    """
    One has tried to solve an empty model.
    """

    def __init__(self) -> None:
        super().__init__("Cannot solve model without any variables.")


class MissingObjectiveError(Exception):
    """
    One has tried to solve an model without any objective.
    """

    def __init__(self) -> None:
        super().__init__("Cannot solve model missing an objective.")
