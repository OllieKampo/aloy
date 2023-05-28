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
    Triangular, trapezoidal, rectangular, piecewise linear, singleton and
    gaussian membership functions
    Trapezoidal resolution hedges
    Rule modules
    Dynamic importance degrees
"""

from abc import ABCMeta, abstractmethod, abstractproperty
from collections import deque
from typing import Callable, Iterable, NamedTuple, Sequence, final
from matplotlib import figure, pyplot as plt

import numpy as np
import numpy.typing as npt

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
        min_val: float,
        max_val: float,
        gain: float = 1.0
    ) -> None:
        """
        Create a proportional fuzzy variable.

        Parameters
        ----------
        `name: str` - The name of the variable.

        `min_val: float` - The minimum value of the variable.

        `max_val: float` - The maximum value of the variable.

        `gain: float = 1.0` - The gain of the variable.
        """
        self.__name: str = name
        self.__min_val: float = min_val
        self.__max_val: float = max_val
        self.__range: float = max_val - min_val
        self.__gain: float = gain

    @property
    def name(self) -> str:
        """Get the name of the variable."""
        return self.__name

    @property
    def max_val(self) -> float:
        """Get the maximum value of the variable."""
        return self.__max_val

    @property
    def min_val(self) -> float:
        """Get the minimum value of the variable."""
        return self.__min_val

    @property
    def value_range(self) -> float:
        """Get the value range of the variable."""
        return self.__range

    @property
    def gain(self) -> float:
        """Get the gain of the variable."""
        return self.__gain

    def get_value(  # pylint: disable=unused-argument
        self,
        error: float,
        delta_time: float,
        abs_tol: float | None
    ) -> float:
        """
        Calculate the weighted normalised proportional value of the variable.

        Parameters
        ----------
        `error: float` - The error of the system.

        `delta_time: float` - The time since the last update.

        `abs_tol: float | None = None` - The absolute tolerance for the
        time difference. If given and not None, if the time difference is
        smaller than the given value, then the integral and derivative
        errors are not updated to avoid precision errors.
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
        min_val: float,
        max_val: float,
        gain: float = 1.0
    ) -> None:
        """
        Create an integral fuzzy variable.

        Parameters
        ----------
        `name: str` - The name of the variable.

        `min_val: float` - The minimum value of the variable.

        `max_val: float` - The maximum value of the variable.

        `gain: float = 1.0` - The gain of the variable.
        """
        super().__init__(name, min_val, max_val, gain)
        self.__center: float = (max_val + min_val) / 2.0
        self.__integral_sum: float = 0.0

    def get_value(
        self,
        error: float,
        delta_time: float,
        abs_tol: float | None
    ) -> float:
        """
        Calculate the estimated weighted normalised integral value of the
        variable.

        Parameters
        ----------
        `error: float` - The error of the system.

        `delta_time: float` - The time since the last update.

        `abs_tol: float | None = None` - The absolute tolerance for the
        time difference. If given and not None, if the time difference is
        smaller than the given value, then the integral and derivative
        errors are not updated to avoid precision errors.
        """
        if delta_time > 0.0 and (abs_tol is None or delta_time > abs_tol):
            self.__integral_sum += (error - self.__center) * delta_time
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
        min_val: float,
        max_val: float,
        gain: float = 1.0,
        average_derivatives: int = 3,
        initial_error: float | None = None
    ) -> None:
        """
        Create a derivative fuzzy variable.

        Parameters
        ----------
        `name: str` - The name of the variable.

        `min_val: float` - The minimum value of the variable.

        `max_val: float` - The maximum value of the variable.

        `gain: float = 1.0` - The gain of the variable.
        """
        super().__init__(name, min_val, max_val, gain)
        self.__latest_error: float | None = None
        self.__derivatives = deque[float](maxlen=average_derivatives)
        self.__initial_error: float | None = initial_error

    def get_value(
        self,
        error: float,
        delta_time: float,
        abs_tol: float | None
    ) -> float:
        """
        Calculate the estimated weighted normalised derivative value of the
        variable.

        Parameters
        ----------
        `error: float` - The error of the system.

        `delta_time: float` - The time since the last update.

        `abs_tol: float | None = None` - The absolute tolerance for the
        time difference. If given and not None, if the time difference is
        smaller than the given value, then the integral and derivative
        errors are not updated to avoid precision errors.
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

    def __init__(self, name: str, *params: float) -> None:
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

    @abstractproperty
    def params(self) -> tuple[float, ...]:
        """Get the parameters of the membership function."""
        raise NotImplementedError

    def __repr__(self) -> str:
        """Get an instantiable string representation of the membership function."""
        return f"{self.__class__.__name__}({self.__name!r}, *{self.params!r})"

    def __str__(self) -> str:
        """Get a huamn readable string representation of the membership function."""
        return f"{self.__class__.__name__} {self.__name!s}: {self.params!s}"

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
        Calculate the activation of a rule using this membership function
        and for the given normalised input variable value and output weight.

        The activation is equal to the degree of truth of the rule (the
        membership of the input value within this membership function)
        multiplied by the output weight of the rule.
        """
        truth: float = self.fuzzify(input_)
        return RuleActivation(truth, truth * output)

    @abstractmethod
    def fuzzify(self, value: float) -> float:
        """
        Fuzzify a value using this membership function.

        This calculates the degree of membership of the given value in this
        membership function.
        """
        ...

    @abstractmethod
    def to_array(
        self
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Convert the membership function to an array.

        This method must be implemented by subclasses.
        """
        ...

    @staticmethod
    def plot_membership_functions(
        membership_functions: Iterable["MembershipFunction"]
    ) -> figure.Figure:
        """
        Plot the given membership functions.

        Parameters
        ----------
        `membership_functions: Iterable[MembershipFunction]` - The membership
        functions to plot.
        """
        fig, ax = plt.subplots()
        for membership_function in membership_functions:
            x, y = membership_function.to_array()
            ax.plot(x, y, label=membership_function.name)
        ax.legend()
        return fig


@final
class TriangularFunction(MembershipFunction):
    """Class defining triangular membership functions."""

    __slots__ = {
        "__start": "The value at which the triangle starts.",
        "__peak": "The value at which the triangle peaks.",
        "__end": "The value at which the triangle ends."
    }

    def __init__(
        self,
        name: str,
        start: float,
        peak: float,
        end: float
    ) -> None:
        """
        Create a triangular membership function.

        A value has a degree of membership of 1.0 if it is equal to the peak.
        Has a degree of membership increasing linearly from 0.0 to 1.0 from the
        start to the peak, and decreasing linearly from 1.0 to 0.0 from the
        peak to the end. Otherwise the degree of membership is 0.0.

        Parameters
        ----------
        `name: str` - The name of the membership function.

        `start: float` - The value at which the triangle starts.

        `peak: float` - The value at which the triangle peaks.

        `end: float` - The value at which the triangle ends.

        Raises
        ------
        `ValueError` - If the start is greater than the peak or the peak is
        greater than the end. Or if any of the parameters are not in the range
        [0.0, 1.0].
        """
        super().__init__(name, start, peak, end)
        if start > peak or peak > end:
            raise ValueError(
                "Invalid triangular membership function parameters."
                "Start must be less than peak and peak less than end."
                f"Got; {start=}, {peak=}, {end=}."
            )
        self.__start: float = start
        self.__peak: float = peak
        self.__end: float = end

    @property
    def params(self) -> tuple[float, float, float]:
        """Get the parameters of the membership function."""
        return self.__start, self.__peak, self.__end

    def fuzzify(self, value: float) -> float:
        """
        Fuzzify a value using this membership function.

        This calculates the degree of membership of the given value in this
        membership function.
        """
        if not (self.__start <= value <= self.__end):
            return 0.0
        if value < self.__peak:
            return (value - self.__start) / (self.__peak - self.__start)
        return (self.__end - value) / (self.__end - self.__peak)

    def to_array(
        self
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Convert the membership function to an array."""
        x_points = np.array([0.0, self.__start, self.__peak, self.__peak, self.__end, 1.0])
        y_points = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
        return (x_points, y_points)


@final
class TrapezoidalFunction(MembershipFunction):
    """Class defining trapezoidal membership functions."""

    __slots__ = {
        "__start": "The value at which the trapezoid starts.",
        "__first_peak": "The value of the first peak of the trapezoid.",
        "__second_peak": "The value of the second peak of the trapezoid.",
        "__end": "The value at which the trapezoid ends."
    }

    def __init__(
        self,
        name: str,
        start: float,
        first_peak: float,
        second_peak: float,
        end: float
    ) -> None:
        """
        Create a trapezoidal membership function.

        A value has a degree of membership of 1.0 if it is between the first
        peak and second peak. Has a degree of membership increasing linearly
        from 0.0 to 1.0 from the start to the first peak, and decreasing
        linearly from 1.0 to 0.0 from the second peak to the end. Otherwise
        the degree of membership is 0.0.

        Parameters
        ----------
        `name: str` - The name of the membership function.

        `start: float` - The value at which the trapezoid starts.

        `first_peak: float` - The value of the first peak of the trapezoid.

        `second_peak: float` - The value of the second peak of the trapezoid.

        `end: float` - The value at which the trapezoid ends.

        Raises
        ------
        `ValueError` - If the start is greater than the first peak, the
        first peak is greater than the second peak or the second peak
        is greater than the end. Or if any of the parameters are not in the
        range [0.0, 1.0].
        """
        super().__init__(name, start, first_peak, second_peak, end)
        if start > first_peak or first_peak > second_peak or second_peak > end:
            raise ValueError(
                "Invalid trapezoidal membership function parameters."
                "Start must be less than first peak, first peak less "
                "than second peak, and second peak less than end."
                f"Got; {start=}, {first_peak=}, {second_peak=}, {end=}."
            )
        self.__start: float = start
        self.__first_peak: float = first_peak
        self.__second_peak: float = second_peak
        self.__end: float = end

    @classmethod
    def create_set(
        cls,
        names: Sequence[str],
        sizes: Sequence[float] | None = None,
        overlap: float | None = None
    ) -> tuple["TrapezoidalFunction", ...]:
        """
        Create a set of trapezoidal membership functions.

        Parameters
        ----------
        `names: Sequence[str]` - The names of the membership functions.

        `sizes: Sequence[float]` - The relative sizes of the membership
        functions. This is the proportion of the total normalised range
        of an input variable that each membership function covers. Must
        sum to 1.0, no value can be 0.0.

        Raises
        ------
        `ValueError` - If the number of names and sizes do not match or the
        sum of the sizes is not 1.0.
        """
        if sizes is not None:
            if len(names) != len(sizes):
                raise ValueError(
                    "The number of names and sizes must match."
                    f"Got; {len(names)=}, {len(sizes)=}."
                )
            if overlap is None:
                raise ValueError(
                    "If sizes is provided, overlap must also be provided."
                )
            if sum(sizes) != 1.0:
                raise ValueError(
                    "The sum of the sizes must be 1.0."
                    f"Got; {sum(sizes)=}."
                )
            if any(size == 0.0 for size in sizes):
                raise ValueError(
                    "No size can be 0.0."
                    f"Got; {sizes=}."
                )
        # Each membership needs its plateau (of the same size) and its ramps
        # (of the same size). The first and last membership functions only
        # have one ramp (right and left respectively).
        membership_functions: list[TrapezoidalFunction] = []
        if sizes is None:
            spaces = np.linspace(0.0, 1.0, len(names) * 2)
            for index, name in enumerate(names, start=-1):
                if index == -1:
                    start = 0.0
                else:
                    start = spaces[(index * 2) + 1]
                first_peak = spaces[(index * 2) + 2]
                second_peak = spaces[(index * 2) + 3]
                if index == len(names) - 2:
                    end = 1.0
                else:
                    end = spaces[(index * 2) + 4]
                membership_functions.append(
                    cls(name, start, first_peak, second_peak, end)
                )
        else:
            space_size = 1.0 / ((len(names) * 2) - 1)
            for index, name in enumerate(names, start=-1):
                start = max((index * 2 * space_size) + space_size, 0.0)
                first_peak = (index * 2 * space_size) + (space_size * 2)
                second_peak = (index * 2 * space_size) + (space_size * 3)
                end = min((index * 2 * space_size) + (space_size * 4), 1.0)
                membership_functions.append(
                    cls(name, start, first_peak, second_peak, end)
                )
        return tuple(membership_functions)

    @property
    def params(self) -> tuple[float, float, float, float]:
        """Get the parameters of the membership function."""
        return (
            self.__start,
            self.__first_peak,
            self.__second_peak,
            self.__end
        )

    def fuzzify(self, value: float) -> float:
        """
        Calculate the degree of membership of the given value in the
        membership function.
        """
        if not (self.__start < value < self.__end):
            return 0.0
        if value < self.__first_peak:
            return (value - self.__start) / (self.__first_peak - self.__start)
        if value < self.__second_peak:
            return 1.0
        return (self.__end - value) / (self.__end - self.__second_peak)

    def to_array(
        self
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Convert the membership function to an array."""
        x_points = np.array(
            [
                0.0,
                self.__start, self.__first_peak,
                self.__second_peak, self.__end,
                1.0
            ]
        )
        y_points = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
        return (x_points, y_points)


@final
class RectangularFunction(MembershipFunction):
    """Class defining rectangular membership functions."""

    __slots__ = {
        "__start": "The value at which the rectangle starts.",
        "__end": "The value at which the rectangle ends."
    }

    def __init__(self, name: str, start: float, end: float) -> None:
        """
        Create a rectangular membership function.

        A value has a degree of membership of 1.0 if it is between the start
        and end values, otherwise it has a membership of 0.0.

        Parameters
        ----------
        `name: str` - The name of the membership function.

        `start: float` - The value at which the rectangle starts.

        `end: float` - The value at which the rectangle ends.

        Raises
        ------
        `ValueError` - If the start is greater than the end. Or if any of the
        parameters are not in the range [0.0, 1.0].
        """
        super().__init__(name, start, end)
        if start > end:
            raise ValueError(
                "Invalid rectangular membership function parameters."
                f"Start must be less than end. Got; {start=}, {end=}."
            )
        self.__start: float = start
        self.__end: float = end

    @property
    def params(self) -> tuple[float, float]:
        """Get the parameters of the membership function."""
        return (self.__start, self.__end)

    def fuzzify(self, value: float) -> float:
        """
        Calculate the degree of membership of the given value in the
        membership function.
        """
        if self.__start <= value <= self.__end:
            return 1.0
        return 0.0

    def to_array(
        self
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Convert the membership function to an array."""
        x_points = np.array(
            [
                0.0,
                self.__start, self.__start,
                self.__end, self.__end,
                1.0
            ]
        )
        y_points = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
        return (x_points, y_points)


@final
class SinusoidalFunction(MembershipFunction):
    """Class defining sinusoidal membership functions."""

    __slots__ = (
        "__start",
        "__end"
    )

    def __init__(self, name: str, start: float, end: float) -> None:
        """Create a sinusoidal membership function."""
        super().__init__(name, start, end)
        if start > end:
            raise ValueError(
                "Invalid sinusoidal membership function parameters."
                f"Start must be less than end. Got; {start=}, {end=}."
            )
        self.__start: float = start
        self.__end: float = end

    @property
    def params(self) -> tuple[float, float]:
        """Get the parameters of the membership function."""
        return (self.__start, self.__end)

    def fuzzify(self, value: float) -> float:
        """
        Calculate the degree of membership of the given value in the
        membership function.
        """
        if not (self.__start <= value <= self.__end):
            return 0.0
        return np.sin(
            ((value - self.__start)
             / (self.__end - self.__start))
            * (np.pi / 2.0)
        )

    def to_array(
        self
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Convert the membership function to an array."""
        x_points = np.linspace(0.0, 1.0, 100)
        y_points = np.sin(x_points * np.pi / 2)
        return (x_points, y_points)


# pylint: disable=abstract-method
class _SaturatedFunction(MembershipFunction):
    """Base class for max- and min-saturated membership functions."""

    __slots__ = ("__scale",)

    def __init__(self, name: str, scale: bool = False) -> None:
        super().__init__(name)
        self.__scale: bool = scale

    @property
    def scale(self) -> bool:
        """
        Whether the degree of membership of the input value is scaled by
        its magnitude.
        """
        return self.__scale


@final
class MaxSaturatedFunction(_SaturatedFunction):
    """Class defining max-saturated membership functions."""

    __slots__ = ()

    # pylint: disable=useless-parent-delegation
    def __init__(self, name: str, scale: bool = False) -> None:
        """
        Create a max-saturated membership function.

        Parameters
        ----------
        `scale: bool = False` - Whether the degree of membership of the input
        value is scaled by its magnitude. If True, a max-saturated input value
        will return the absolute value as output. Otherwise, a max-saturated
        input value will always return 1.0.
        """
        super().__init__(name, scale)

    def fuzzify(self, value: float) -> float:
        """
        Calculate the degree of membership of the given value in the
        membership function.
        """
        return (abs(value) if self.scale else 1.0) if value >= 1.0 else 0.0

    def to_array(
        self
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Convert the membership function to an array."""
        x_points = np.array([0.0, 1.0, 1.0, 1.1])
        y_points = np.array([0.0, 0.0, 1.0, 1.0])
        return (x_points, y_points)


@final
class MinSaturatedFunction(_SaturatedFunction):
    """Class defining min-saturated membership functions."""

    __slots__ = ()

    # pylint: disable=useless-parent-delegation
    def __init__(self, name: str, scale: bool = False) -> None:
        """
        Create a min-saturated membership function.

        Parameters
        ----------
        `scale: bool = False` - Whether the degree of membership of the input
        value is scaled by its magnitude. If True, a min-saturated input value
        will return the absolute value as output. Otherwise, a min-saturated
        input value will always return 1.0.
        """
        super().__init__(name, scale)

    def fuzzify(self, value: float) -> float:
        """
        Calculate the degree of membership of the given value in the
        membership function.
        """
        return (abs(value) if self.scale else 1.0) if value <= 1.0 else 0.0

    def to_array(
        self
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Convert the membership function to an array."""
        x_points = np.array([-0.1, 0.0, 0.0, 1.0])
        y_points = np.array([1.0, 1.0, 0.0, 0.0])
        return (x_points, y_points)


class FuzzyController(Controller):
    """Class defining a fuzzy controller."""

    __slots__ = {
        "__variables": "Fuzzy variables in the controller.",
        "__mem_funcs": "Membership functions in the controller.",
        "__rules": "Rules in the controller."
    }

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

    # tiny = TrapezoidalFunction("tiny", 0.0, 0.0, 0.1818, 0.2727)
    # small = TrapezoidalFunction("small", 0.1818, 0.2727, 0.4545, 0.5454)
    # big = TrapezoidalFunction("big", 0.4545, 0.5454, 0.7272, 0.8181)
    # large = TrapezoidalFunction("large", 0.7272, 0.8181, 1.0, 1.0)

    tiny, small, big, large = TrapezoidalFunction.create_set(
        ["tiny", "small", "big", "large"],
        [0.25, 0.25, 0.25, 0.25]
    )
    for mem_func in [tiny, small, big, large]:
        print(mem_func.params)
    fig = MembershipFunction.plot_membership_functions(
        [tiny, small, big, large]
    )
    fig.show()
    input()

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
