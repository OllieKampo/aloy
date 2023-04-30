###########################################################################
###########################################################################
## Fuzzy controllers.                                                    ##
##                                                                       ##
## Copyright (C)  2022  Oliver Michael Kamperis                          ##
## Email: o.m.kamperis@gmail.com                                         ##
##                                                                       ##
## This program is free software: you can redistribute it and/or modify  ##
## it under the terms of the GNU General Public License as published by  ##
## the Free Software Foundation, either version 3 of the License, or     ##
## any later version.                                                    ##
##                                                                       ##
## This program is distributed in the hope that it will be useful,       ##
## but WITHOUT ANY WARRANTY; without even the implied warranty of        ##
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          ##
## GNU General Public License for more details.                          ##
##                                                                       ##
## You should have received a copy of the GNU General Public License     ##
## along with this program. If not, see <https://www.gnu.org/licenses/>. ##
###########################################################################
###########################################################################

"""
Module defining Takagi-Seguno style fuzzy controllers.

Contains inbuilt functionality for the following:
    Proportional, integral and differential input variables
    Triangular, trapezoidal, rectangular, piecewise linear, singleton and gaussian membership functions
    Trapezoidal resolution hedges
    Rule modules
    Dynamic importance degrees
"""

from abc import ABCMeta, abstractmethod
from collections import deque
from numbers import Real
import numpy as np
import numpy.typing as npt
from typing import Callable, Iterable, NamedTuple, final
from control.controllers import Controller, calc_error, clamp


class FuzzyVariable:
    """Class defining proportional fuzzy input variables."""

    __slots__ = {
        "__name": "The name of the variable.",
        "__min_val": "The minimum value of the variable.",
        "__max_val": "The maximum value of the variable.",
        "__range": "The value range of the variable.",
        "__gain": "The gain of the variable."
    }

    def __init__(
        self,
        name: str,
        min_val: Real,
        max_val: Real,
        gain: float = 1.0
    ) -> None:
        """
        Create a proportional fuzzy variable.

        Parameters
        ----------
        `name: str` - The name of the variable.

        `min_val: Real` - The minimum value of the variable.

        `max_val: Real` - The maximum value of the variable.

        `gain: float` - The gain of the variable.
        """
        self.__name: str = name
        self.__min_val: Real = min_val
        self.__max_val: Real = max_val
        self.__range: Real = max_val - min_val
        self.__gain: float = gain

    @property
    def name(self) -> str:
        """Get the name of the variable."""
        return self.__name

    @property
    def max_val(self) -> Real:
        """Get the maximum value of the variable."""
        return self.__max_val

    @property
    def min_val(self) -> Real:
        """Get the minimum value of the variable."""
        return self.__min_val

    @property
    def value_range(self) -> Real:
        """Get the value range of the variable."""
        return self.__range

    @property
    def gain(self) -> float:
        """Get the gain of the variable."""
        return self.__gain

    @final
    def __hash__(self) -> int:
        """Get the hash of the variable."""
        return hash(self.__name)

    def get_value(
        self,
        error: Real,
        delta_time: float,
        abs_tol: float
    ) -> float:
        """
        Calculate the weighted normalised proportional value of the variable.
        """
        return ((self.__gain * error) - self.__min_val) / self.__range

    def reset(self) -> None:
        """Reset the variable."""
        pass


class IntegralFuzzyVariable(FuzzyVariable):
    """
    Class defining integral fuzzy input variables.

    An integral fuzzy input variable attempts to estimate the
    integral (trapezoidal sum over time) of the error of the system.
    """

    __slots__ = {
        "__center": "The center of the variable.",
        "__integral_sum": "The integral sum of the variable."
    }

    def __init__(
        self,
        name: str,
        min_val: Real,
        max_val: Real,
        gain: float = 1.0
    ) -> None:
        """Create an integral fuzzy variable."""
        super().__init__(name, min_val, max_val, gain)
        self.__center: Real = (max_val + min_val) / 2.0
        self.__integral_sum: Real = 0.0

    def get_value(
        self,
        error: Real,
        delta_time: float,
        abs_tol: float
    ) -> float:
        """
        Calculate the estimated weighted normalised integral value of the
        variable.
        """
        if delta_time > 0.0 and (abs_tol is None or delta_time > abs_tol):
            self.__integral_sum += (error - self.__center) * delta_time
            print(f"Integral sum: {self.__integral_sum}")
        return (((self.gain * self.__integral_sum) + self.__center)
                - self.min_val) / self.value_range

    def reset(self) -> None:
        """Reset the variable."""
        self.__integral_sum = 0.0


class DerivativeFuzzyVariable(FuzzyVariable):
    """
    Class defining derivative fuzzy input variables.

    A derivative fuzzy input variable attempts to estimate the
    derivative (rate of change) of the error of the system.
    """

    __slots__ = {
        "__initial_error": "Initial error of the variable.",
        "__latest_error": "Latest error of the variable.",
        "__derivatives": "Previous derivatives of the variable."
    }

    def __init__(
        self,
        name: str,
        min_val: Real,
        max_val: Real,
        gain: float = 1.0,
        average_derivatives: int = 3,
        initial_error: float | None = None
    ) -> None:
        """Create a derivative fuzzy variable."""
        super().__init__(name, min_val, max_val, gain)
        self.__latest_error: float | None = None
        self.__derivatives = deque[float](maxlen=average_derivatives)
        self.__initial_error: float | None = initial_error

    def get_value(
        self,
        error: Real,
        delta_time: float,
        abs_tol: float
    ) -> float:
        """
        Calculate the estimated weighted normalised derivative value of the
        variable.
        """
        derivative: float = 0.0
        if delta_time > 0.0 and (abs_tol is None or delta_time > abs_tol):
            if self.__latest_error is not None:
                derivative = (error - self.__latest_error) / delta_time
                self.__derivatives.append(derivative)
                derivative = sum(self.__derivatives) / len(self.__derivatives)
        self.__latest_error = error
        return ((self.gain * derivative) - self.min_val) / self.value_range

    def reset(self) -> None:
        """Reset the variable."""
        self.__latest_error = self.__initial_error
        self.__derivatives.clear()


class RuleActivation(NamedTuple):
    """
    Class defining the activation of a fuzzy rule.

    Items
    -----
    `truth : float` - The degree of truth of the rule.

    `activation : float` - The degree of activation of the rule.
    This is the weighted output of the rule, given its truth.
    """

    truth: float
    activation: float


class MembershipFunction(metaclass=ABCMeta):
    """Base class for fuzzy set membership functions."""

    __slots__ = {
        "__name": "The name of the membership function."
    }

    def __init__(self, name: str, *params: Real) -> None:
        """Create a membership function."""
        if any((not 0.0 <= param <= 1.0) for param in params):
            raise ValueError(
                "Membership function parameters must be between 0.0 and 1.0."
                f"Got; {params!r}."
            )
        self.__name: str = name

    @property
    def name(self) -> str:
        """Get the name of the variable."""
        return self.__name

    @final
    def __hash__(self) -> int:
        """Get the hash of the membership function."""
        return hash(self.__name)

    def get_activation(
        self,
        input_: float,
        output: float
    ) -> RuleActivation:
        """
        Calculate the activation of the rule.

        The activation is equal to the degree of truth of the rule
        multiplied by the output of the rule. Where the truth is
        equivalent to the degree of membership of the value of
        its given error variable in its membership function.
        """
        truth: float = self.fuzzify(input_)
        return RuleActivation(truth, truth * output)

    @abstractmethod
    def fuzzify(self, value: Real) -> Real:
        """
        Fuzzify a value.

        This method must be implemented by subclasses.
        """
        ...

    @abstractmethod
    def to_array(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Convert the membership function to an array.

        This method must be implemented by subclasses.
        """
        ...


@final
class TriangularFunction(MembershipFunction):
    """Class defining triangular membership functions."""

    __slots__ = ("__start",
                 "__peak",
                 "__end")

    def __init__(self, name: str, start: Real, peak: Real, end: Real) -> None:
        """Create a triangular membership function."""
        super().__init__(name, start, peak, end)
        if start > peak or peak > end:
            raise ValueError("Invalid triangular membership function parameters."
                             "Start must be less than peak and peak less than end."
                             f"Got; {start=}, {peak=}, {end=}.")
        self.__start: Real = start
        self.__peak: Real = peak
        self.__end: Real = end

    def fuzzify(self, value: float) -> float:
        """Calculate the degree of membership of the given value in the membership function."""
        if not (self.__start <= value <= self.__end):
            return 0.0
        if value < self.__peak:
            return (value - self.__start) / (self.__peak - self.__start)
        return (self.__end - value) / (self.__end - self.__peak)

    def to_array(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Convert the membership function to an array."""
        x_points = np.array([0.0, self.__start, self.__peak, self.__end, 1.0])
        y_points = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
        return (x_points, y_points)


@final
class TrapezoidalFunction(MembershipFunction):
    """Class defining trapezoidal membership functions."""

    __slots__ = ("__start",
                 "__first_peak",
                 "__second_peak",
                 "__end")

    def __init__(self, name: str, start: Real, first_peak: Real, second_peak: Real, end: Real) -> None:
        """Create a trapezoidal membership function."""
        super().__init__(name, start, first_peak, second_peak, end)
        if start > first_peak or first_peak > second_peak or second_peak > end:
            raise ValueError("Invalid trapezoidal membership function parameters."
                             "Start must be less than first peak, first peak less "
                             "than second peak and second peak less than end."
                             f"Got; {start=}, {first_peak=}, {second_peak=}, {end=}.")
        self.__start: Real = start
        self.__first_peak: Real = first_peak
        self.__second_peak: Real = second_peak
        self.__end: Real = end

    def fuzzify(self, value: float) -> float:
        """Calculate the degree of membership of the given value in the membership function."""
        if not (self.__start < value < self.__end):
            return 0.0
        if value < self.__first_peak:
            return (value - self.__start) / (self.__first_peak - self.__start)
        if value < self.__second_peak:
            return 1.0
        return (self.__end - value) / (self.__end - self.__second_peak)

    def to_array(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Convert the membership function to an array."""
        x_points = np.array([0.0, self.__start, self.__first_peak, self.__second_peak, self.__end, 1.0])
        y_points = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
        return (x_points, y_points)


@final
class RectangularFunction(MembershipFunction):
    """Class defining rectangular membership functions."""

    __slots__ = ("__start",
                 "__end")

    def __init__(self, name: str, start: Real, end: Real) -> None:
        """Create a rectangular membership function."""
        super().__init__(name, start, end)
        if start > end:
            raise ValueError("Invalid rectangular membership function parameters."
                             f"Start must be less than end. Got; {start=}, {end=}.")
        self.__start: Real = start
        self.__end: Real = end

    def fuzzify(self, value: float) -> float:
        """Calculate the degree of membership of the given value in the membership function."""
        if self.__start <= value <= self.__end:
            return 1.0
        return 0.0

    def to_array(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Convert the membership function to an array."""
        x_points = np.array([0.0, self.__start, self.__start, self.__end, self.__end, 1.0])
        y_points = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
        return (x_points, y_points)


@final
class SinusoidalFunction(MembershipFunction):
    """Class defining sinusoidal membership functions."""

    __slots__ = ("__start",
                 "__end")

    def __init__(self, name: str, start: Real, end: Real) -> None:
        """Create a sinusoidal membership function."""
        super().__init__(name, start, end)
        if start > end:
            raise ValueError("Invalid sinusoidal membership function parameters."
                             f"Start must be less than end. Got; {start=}, {end=}.")
        self.__start: Real = start
        self.__end: Real = end

    def fuzzify(self, value: float) -> float:
        """Calculate the degree of membership of the given value in the membership function."""
        if not (self.__start <= value <= self.__end):
            return 0.0
        return np.sin((value - self.__start) / (self.__end - self.__start) * np.pi / 2)

    def to_array(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Convert the membership function to an array."""
        x_points = np.linspace(0.0, 1.0, 100)
        y_points = np.sin(x_points * np.pi / 2)
        return (x_points, y_points)


class _SaturatedFunction(MembershipFunction):
    """Base class for max- and min-saturated membership functions."""

    __slots__ = ("__scale")

    def __init__(self, name: str, scale: bool = False) -> None:
        super().__init__(name)
        self.__scale: bool = scale

    @property
    def scale(self) -> bool:
        """Whether the degree of membership of the input value is scaled by its magnitude."""
        return self.__scale


@final
class MaxSaturatedFunction(_SaturatedFunction):
    """Class defining max-saturated membership functions."""

    __slots__ = ()

    def __init__(self, name: str, scale: bool = False) -> None:
        """
        Create a max-saturated membership function.

        Parameters
        ----------
        `scale : bool = False` - Whether the degree of membership of the input value is scaled by its magnitude.
        If true, a max-saturated input value will return the absolute value as output.
        Otherwise, a max-saturated input value will always return 1.0.
        """
        super().__init__(name, scale)

    def fuzzify(self, value: float) -> float:
        """Calculate the degree of membership of the given value in the membership function."""
        return (abs(value) if self.scale else 1.0) if value >= 1.0 else 0.0

    def to_array(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Convert the membership function to an array."""
        x_points = np.array([0.0, 1.0, 1.0, 1.1])
        y_points = np.array([0.0, 0.0, 1.0, 1.0])
        return (x_points, y_points)


@final
class MinSaturatedFunction(_SaturatedFunction):
    """Class defining min-saturated membership functions."""

    __slots__ = ()

    def __init__(self, name: str, scale: bool = False) -> None:
        """
        Create a min-saturated membership function.

        Parameters
        ----------
        `scale : bool = False` - Whether the degree of membership of the input
        value is scaled by its magnitude. If True, a min-saturated input value
        will return the absolute value as output. Otherwise, a min-saturated
        input value will always return 1.0.
        """
        super().__init__(name, scale)

    def fuzzify(self, value: float) -> float:
        """Calculate the degree of membership of the given value in the membership function."""
        return (abs(value) if self.scale else 1.0) if value <= 1.0 else 0.0

    def to_array(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Convert the membership function to an array."""
        x_points = np.array([-0.1, 0.0, 0.0, 1.0])
        y_points = np.array([1.0, 1.0, 0.0, 0.0])
        return (x_points, y_points)


class FuzzyController(Controller):

    __slots__ = (
        "__variables",
        "__mem_funcs",
        "__rules"
    )

    def __init__(
        self,
        variables: Iterable[FuzzyVariable],
        mem_funcs: Iterable[MembershipFunction],
        rules: Iterable[tuple[str, str, float]] | None = None,
        input_limits: tuple[float | None, float | None] = (None, None),
        output_limits: tuple[float | None, float | None] = (None, None),
        input_trans: Callable[[float], float] | None = None,
        error_trans: Callable[[float], float] | None = None,
        output_trans: Callable[[float], float] | None = None,
        initial_error: float | None = None
    ) -> None:
        """Create a fuzzy controller."""
        super().__init__(
            input_limits,
            output_limits,
            input_trans,
            error_trans,
            output_trans,
            initial_error
        )

        self.__variables: dict[str, FuzzyVariable] = {
            var.name: var for var in variables
        }
        self.__mem_funcs: dict[str, MembershipFunction] = {
            mem_func.name: mem_func for mem_func in mem_funcs
        }
        self.__rules: dict[FuzzyVariable, dict[MembershipFunction, float]] = {}
        if rules is not None:
            for rule in rules:
                self.add_rule(*rule)

    def __str__(self) -> str:
        """
        Return a human-readable string representation of the fuzzy controller.
        """
        return f"Fuzzy Controller: total rules = {len(self.__rules)}"

    def __repr__(self) -> str:
        """Return a parseable string representation of the fuzzy controller."""
        rules: list[tuple[str, str, float]] = [
            (var.name, mem_func.name, output)
            for var in self.__rules
            for mem_func, output in self.__rules[var].items()
        ]
        return (
            "FuzzyController("
            f"variables={self.__variables}, "
            f"mem_funcs={self.__mem_funcs}, "
            f"rules={rules}, "
            f"input_limits={self.input_limits}, "
            f"output_limits={self.output_limits}, "
            f"input_transform={self.input_transform}, "
            f"error_transform={self.input_transform}, "
            f"output_transform={self.output_transform}, "
            f"initial_error={self.initial_error})"
        )

    def add_variable(self, var: FuzzyVariable) -> None:
        """
        Add a variable to the controller.

        Parameters
        ----------
        `var: FuzzyVariable` - The variable to add to the controller.
        """
        if var.name in self.__variables:
            raise KeyError(f"Variable with name {var.name} already exists.")
        self.__variables[var.name] = var

    def get_variable(self, name: str) -> FuzzyVariable:
        """
        Get the variable with the given name.

        Parameters
        ----------
        `name: str` - The name of the variable to get.

        Returns
        -------
        `FuzzyVariable` - The variable with the given name.
        """
        return self.__variables[name]

    def remove_variable(self, name: str) -> None:
        """
        Remove the variable with the given name.

        Parameters
        ----------
        `name: str` - The name of the variable to remove.
        """
        del self.__variables[name]

    def add_mem_func(self, mem_func: MembershipFunction) -> None:
        """
        Add a membership function to the controller.

        Parameters
        ----------
        `mem_func: MembershipFunction` - The membership function to add to the
        controller.
        """
        if mem_func.name in self.__mem_funcs:
            raise KeyError(f"Membership function with name {mem_func.name} "
                           "already exists.")
        self.__mem_funcs[mem_func.name] = mem_func

    def get_mem_func(self, name: str) -> MembershipFunction:
        """
        Get the membership function with the given name.

        Parameters
        ----------
        `name: str` - The name of the membership function to get.

        Returns
        -------
        `MembershipFunction` - The membership function with the given name.
        """
        return self.__mem_funcs[name]

    def remove_mem_func(self, name: str) -> None:
        """
        Remove the membership function with the given name.

        Parameters
        ----------
        `name: str` - The name of the membership function to remove.
        """
        del self.__mem_funcs[name]

    def add_rule(self, var: str, mem_func: str, output: float) -> None:
        """
        Add a rule to the controller.

        Parameters
        ----------
        `var: str` - The name of the variable to which the rule applies.

        `mem_func: str` - The name of the membership function to which the
        rule applies.

        `output: float` - The output of the rule.
        """
        fuzzy_var: FuzzyVariable = self.__variables[var]
        fuzzy_mem_func: MembershipFunction = self.__mem_funcs[mem_func]
        if fuzzy_var not in self.__rules:
            self.__rules[fuzzy_var] = {}
        if fuzzy_mem_func in self.__rules[fuzzy_var]:
            raise KeyError(f"Rule for {var} and {mem_func} already exist.")
        self.__rules[fuzzy_var][fuzzy_mem_func] = output

    def update_rule(self, var: str, mem_func: str, output: float) -> None:
        """
        Update a rule in the controller.

        Parameters
        ----------
        `var: str` - The name of the variable to which the rule applies.

        `mem_func: str` - The name of the membership function to which the
        rule applies.

        `output: float` - The output of the rule.

        Raises
        ------
        `KeyError` - If the variable or membership function does not exist, or
        if the rule does not exist.
        """
        fuzzy_var: FuzzyVariable = self.__variables[var]
        fuzzy_mem_func: MembershipFunction = self.__mem_funcs[mem_func]
        if (fuzzy_var not in self.__rules or
                fuzzy_mem_func not in self.__rules[fuzzy_var]):
            raise KeyError(f"Rule for {var} and {mem_func} does not exist.")
        self.__rules[fuzzy_var][fuzzy_mem_func] = output

    def remove_rule(self, var: str, mem_func: str) -> None:
        """
        Remove a rule from the controller.

        Parameters
        ----------
        `var: str` - The name of the variable to which the rule applies.

        `mem_func: str` - The name of the membership function to which the
        rule applies.

        Raises
        ------
        `KeyError` - If the variable or membership function does not exist, or
        if the rule does not exist.
        """
        fuzzy_var: FuzzyVariable = self.__variables[var]
        fuzzy_mem_func: MembershipFunction = self.__mem_funcs[mem_func]
        if (fuzzy_var not in self.__rules or
                fuzzy_mem_func not in self.__rules[fuzzy_var]):
            raise KeyError(f"Rule for {var} and {mem_func} does not exist.")
        del self.__rules[fuzzy_var][fuzzy_mem_func]
        if not self.__rules[fuzzy_var]:
            del self.__rules[fuzzy_var]

    def control_output(
        self,
        control_input: float,
        setpoint: float, /,
        delta_time: float,
        abs_tol: float | None = None
    ) -> float:
        if delta_time < 0.0:
            raise ValueError("The time difference must be positive. "
                             f"Got; {delta_time}.")

        # Calculate control input and error.
        control_input = self.transform_input(control_input)
        control_error: float = calc_error(control_input, setpoint,
                                          *self.input_limits)
        control_error = self.transform_error(control_error)

        # Calculate the value of each variable.
        var_inputs: dict[str, float] = {
            var.name: var.get_value(control_error, delta_time, abs_tol)
            for var in self.__rules
        }

        truth_sum: dict[str, float] = {
            var.name: 0.0 for var in self.__rules
        }
        activation_sum: dict[str, float] = truth_sum.copy()

        # Calculate the degree of truth and activation of each rule;
        # the weighted average of the activation of each rule.
        for var, rules in self.__rules.items():
            for mem_func, output in rules.items():
                activation: RuleActivation = mem_func.get_activation(
                    var_inputs[var.name], output
                )
                truth_sum[var.name] = truth_sum[var.name] + activation.truth
                activation_sum[var.name] = (
                    activation_sum[var.name] + activation.activation
                )

        # Calculate control output (defuzzification);
        # the sum over all variables of the sum of the activation of each rule
        # divided by the sum of the degree of truth of each rule.
        control_output: float = sum(
            (activation_sum[var.name] / truth_sum[var.name])
            if truth_sum[var.name] != 0.0 else 0.0
            for var in self.__rules
        )
        print(activation_sum)
        print(truth_sum)
        control_output = clamp(self.transform_output(control_output),
                               *self.output_limits)

        self._latest_input = control_input    # type: ignore
        self._latest_error = control_error    # type: ignore
        self._latest_output = control_output  # type: ignore

        return control_output

    def reset(self) -> None:
        """Reset the controller."""
        super().reset()
        for var in self.__variables.values():
            var.reset()


if __name__ == "__main__":
    proportional = FuzzyVariable("proportional", -10.0, 10.0, gain=1.0)
    derivative = DerivativeFuzzyVariable("derivative", -10.0, 10.0, gain=0.0)
    integral = IntegralFuzzyVariable("integral", -10.0, 10.0, gain=1.0)

    tiny = TrapezoidalFunction("tiny", 0.0, 0.0, 0.1818, 0.2727)
    small = TrapezoidalFunction("small", 0.1818, 0.2727, 0.4545, 0.5454)
    big = TrapezoidalFunction("big", 0.4545, 0.5454, 0.7272, 0.8181)
    large = TrapezoidalFunction("large", 0.7272, 0.8181, 1.0, 1.0)

    controller = FuzzyController(
        [proportional, derivative, integral],
        [tiny, small, big, large]
    )
    controller.add_rule("proportional", "large", 1.0)
    controller.add_rule("proportional", "big", 0.5)
    controller.add_rule("proportional", "small", -0.5)
    controller.add_rule("proportional", "tiny", -1.0)
    controller.add_rule("derivative", "large", 1.0)
    controller.add_rule("derivative", "big", 0.5)
    controller.add_rule("derivative", "small", -0.5)
    controller.add_rule("derivative", "tiny", -1.0)
    controller.add_rule("integral", "large", 1.0)
    controller.add_rule("integral", "big", 0.5)
    controller.add_rule("integral", "small", -0.5)
    controller.add_rule("integral", "tiny", -1.0)

    for i in range(100):
        print(controller.control_output(1.0, 0.0, 0.1))
        # print(controller.control_output(((2.0 / 100.0) * i) - 1.0, 0.0, 0.25))
    controller.reset()
