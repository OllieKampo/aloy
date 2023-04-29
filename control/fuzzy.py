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

"""Module defining fuzzy controllers."""

from abc import ABCMeta, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass
from numbers import Real
import time
import numpy as np
import numpy.typing as npt
from typing import Iterable, NamedTuple, final
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
        Calculate the estimated weighted normalised proportional value of the
        variable.
        """
        return ((self.__gain * error) - self.__min_val) / self.__range


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

    def get_value(self, error: Real, delta_time: float, abs_tol: float) -> Real:
        """Calculate the estimated weighted normalised integral value of the variable."""
        self.__integral_sum += (error - self.__center) * delta_time
        return (((self.gain * self.__integral_sum) + self.__center) - self.min_val) / self.value_range


class DerivativeFuzzyVariable(FuzzyVariable):
    """
    Class defining derivative fuzzy input variables.

    A derivative fuzzy input variable attempts to estimate the
    derivative (rate of change) of the error of the system.
    """

    __slots__ = (
        "__last_error",
        "__previous_derivatives",
        "__time_point"
    )

    def __init__(
        self,
        name: str,
        min_val: Real,
        max_val: Real,
        gain: float = 1.0,
        average_derivatives: int = 3
    ) -> None:
        """Create a derivative fuzzy variable."""
        super().__init__(name, min_val, max_val, gain)
        self.__previous_error: float | None = None
        self.__previous_derivatives: deque[float] = deque(maxlen=average_derivatives)

    def get_value(self, input: Real) -> Real:
        """Calculate the estimated weighted normalised derivative value of the variable."""
        time_now: float = time.perf_counter()
        if self.__previous_error is not None:
            self.__previous_derivatives.append((input - self.__previous_error) / (time_now - self.__time_point))
        self.__previous_error = input
        self.__time_point = time_now
        return ((self.gain * (sum(self.__previous_derivatives) / len(self.__previous_derivatives))) - self.__min_val) / self.value_range


class MembershipFunction(metaclass=ABCMeta):
    """Base class for fuzzy set membership functions."""

    __slots__ = ("__name")

    def __init__(self, name: str, *params: Real) -> None:
        """Create a membership function."""
        if any((not 0.0 <= param <= 1.0) for param in params):
            raise ValueError("Membership function parameters must be between 0.0 and 1.0.")
        self.__name: str = name

    @final
    def __hash__(self) -> int:
        """Get the hash of the membership function."""
        return hash(self.__name)

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
        if not (self.__start <= value <= self.__end):
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


@dataclass
class FuzzyRule(NamedTuple):
    """Class defining fuzzy rules."""

    mem_func: MembershipFunction
    output: float

    def get_activation(self, input: float) -> RuleActivation:
        """
        Calculate the activation of the rule.

        The activation is equal to the degree of truth of the rule
        multiplied by the output of the rule. Where the truth is
        equivalent to the degree of membership of the value of
        its given error variable in its membership function.
        """
        truth: float = self.mem_func.fuzzify(input)
        return RuleActivation(truth, truth * self.output)


class FuzzyController(Controller):

    __slots__ = (
        "__variables",
        "__mem_funcs",
        "__rules",
    )

    def __init__(
        self,
        variables: Iterable[FuzzyVariable],
        mem_funcs: Iterable[MembershipFunction]
    ) -> None:
        """Create a fuzzy controller."""
        self.__variables = set[FuzzyVariable](variables)
        self.__mem_funcs = set[MembershipFunction](mem_funcs)
        self.__rules: dict[FuzzyVariable, FuzzyRule] = {}

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
            for var in self.__variables
        }

        truth_sum = defaultdict(float)
        activation_sum = defaultdict(float)

        # Calculate the degree of truth and activation of each rule.
        for var, rule in self.__rules.items():
            activation: RuleActivation = rule.get_activation(
                var_inputs[var.name]
            )
            truth_sum[var.name] = truth_sum[var.name] + activation.truth
            activation_sum[var.name] = (
                activation_sum[var.name] + activation.activation
            )

        # Calculate control output.
        control_output: float = sum(
            activation_sum[var.name] / truth_sum[var.name]
            for var in self.__variables
        )
        control_output = clamp(self.transform_output(control_output),
                               *self.output_limits)

        self._latest_input = control_input    # type: ignore
        self._latest_error = control_error    # type: ignore
        self._latest_output = control_output  # type: ignore

        return control_output
