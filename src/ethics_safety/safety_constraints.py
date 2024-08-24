# BeSAI/src/ethics_safety/safety_constraints.py

import functools
from typing import Any, Callable, Dict

class SafetyConstraint:
    def __init__(self, name: str, check_function: Callable[[Any], bool], error_message: str):
        self.name = name
        self.check_function = check_function
        self.error_message = error_message

class SafetySystem:
    def __init__(self):
        self.constraints: Dict[str, SafetyConstraint] = {}

    def add_constraint(self, constraint: SafetyConstraint):
        self.constraints[constraint.name] = constraint

    def check_safety(self, action: Any) -> bool:
        for constraint in self.constraints.values():
            if not constraint.check_function(action):
                print(f"Safety violation: {constraint.error_message}")
                return False
        return True

    def safe_execution(self, func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            action = {"function": func.__name__, "args": args, "kwargs": kwargs}
            if self.check_safety(action):
                return func(*args, **kwargs)
            else:
                raise SafetyViolationError("Safety check failed")
        return wrapper

class SafetyViolationError(Exception):
    pass

# Example usage
safety_system = SafetySystem()

# Example constraint: Prevent division by zero
def no_division_by_zero(action: Dict[str, Any]) -> bool:
    if action["function"] == "divide" and action["args"][1] == 0:
        return False
    return True

safety_system.add_constraint(SafetyConstraint(
    "no_division_by_zero",
    no_division_by_zero,
    "Cannot divide by zero"
))

@safety_system.safe_execution
def divide(a: float, b: float) -> float:
    return a / b

# Test the safety system
try:
    result = divide(10, 2)
    print(f"10 / 2 = {result}")
    result = divide(10, 0)
    print(f"10 / 0 = {result}")  # This should raise a SafetyViolationError
except SafetyViolationError as e:
    print(f"Caught safety violation: {e}")
