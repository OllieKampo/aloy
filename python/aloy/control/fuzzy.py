###############################################################################
# Copyright (C) 2023 Oliver Michael Kamperis
# Email: olliekampo@gmail.com
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for details.
#
# You should have received a copy of the GNU General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>.

"""
Module defining Takagi-Seguno style fuzzy controllers.

Contains inbuilt functionality for the following:
    Proportional, integral and differential input variables
    Triangular, trapezoidal, rectangular, and sinusoidal membership functions
    Resolution hedges
    Rule modules
"""

from abc import ABCMeta, abstractmethod
from collections import deque
from typing import Callable, Iterable, NamedTuple, Sequence, final
from matplotlib import figure, pyplot as plt

import numpy as np
import numpy.typing as npt

from aloy.control.controllers import Controller, calc_error, clamp

__copyright__ = "Copyright (C) 2023 Oliver Michael Kamperis"
__license__ = "GPL-3.0"
__version__ = "1.0.1"

__all__ = (
    "FuzzyVariable",
    "IntegralFuzzyVariable",
    "DerivativeFuzzyVariable",
    "RuleActivation",
    "MembershipFunction",
    "TriangularFunction",
    "TrapezoidalFunction",
    "RectangularFunction",
    "SinusoidalFunction",
    "MaxSaturatedFunction",
    "MinSaturatedFunction",
    "create_membership_function_set",
    "plot_membership_functions",
    "FuzzyControllerTerms",
    "FuzzyController"
)


def __dir__() -> tuple[str, ...]:
    """Get the names of module attributes."""
    return __all__


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
        return None


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
        super().__init__(
            name=name,
            min_val=min_val,
            max_val=max_val,
            gain=gain
        )
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
        super().__init__(
            name=name,
            min_val=min_val,
            max_val=max_val,
            gain=gain
        )
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
    `truth: float` - The degree of truth of the rule.

    `activation: float` - The degree of activation of the rule.
    This is the weighted output of the rule, given its truth.
    """

    truth: float
    activation: float


class MembershipFunction(metaclass=ABCMeta):
    """
    Base class for fuzzy set membership functions.

    A membership function is a function that maps the value of an input
    variable to a degree of membership of that value within a fuzzy set.
    The degree of membership is a value between 0.0 and 1.0, where 0.0
    represents no membership (false), 1.0 represents full membership
    (true), and values between 0.0 and 1.0 represent partial membership
    (partially true).
    """

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

    @property
    @abstractmethod
    def params(self) -> tuple[float, ...]:
        """Get the parameters of the membership function."""
        raise NotImplementedError

    def __repr__(self) -> str:
        """
        Get an instantiable string representation of the membership function.
        """
        return f"{self.__class__.__name__}({self.__name!r}, *{self.params!r})"

    def __str__(self) -> str:
        """
        Get a human readable string representation of the membership function.
        """
        return f"{self.__class__.__name__} {self.__name!s}: {self.params!s}"

    @final
    def __hash__(self) -> int:
        """Get the hash of the membership function."""
        return hash(self.__name)

    @final
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

        Parameters
        ----------
        `input_: float` - The normalised input variable value.

        `output: float` - The output weight of the rule.

        Returns
        -------
        `RuleActivation` - The activation of the rule.
        See `aloy.control.fuzzy.RuleActivation` for details.
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
        x_points = np.array([0.0, self.__start, self.__peak, self.__end, 1.0])
        y_points = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
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
    def params(self) -> tuple:
        """Get the parameters of the membership function."""
        return ()

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


def create_membership_function_set(
    names: Sequence[str],
    sizes: Sequence[float] | None = None,
    overlap: float = 0.1,
    include_saturated: bool | list[str] = False,
    scale_saturated: bool = False
) -> tuple["TrapezoidalFunction", ...]:
    """
    Create a set of trapezoidal (and possibly saturated) membership functions.

    Parameters
    ----------
    `names: Sequence[str]` - The names of the membership functions.

    `sizes: Sequence[float] | None = None` - The relative sizes of the
    membership functions. This is the proportion of the total normalised range
    of an input variable that each membership function covers. Must sum to 1.0,
    no value can be 0.0. If not given or None, all membership functions will
    have the same size and be evenly spaced across the normalised range.

    `overlap: float = 0.1` - The amount of overlap between membership
    functions. Must be less than the size of all membership functions.

    `include_saturated: bool = False` - Whether to include max- and
    min-saturated membership functions. By default, the saturated membership
    functions are named 'max-saturated' and 'min-saturated'. If a list of
    strings is given, then the first and seconds strings are used as the names
    of the max- and min-saturated membership functions respectively.

    `scale_saturated: bool = False` - Whether the degree of membership of the
    input value of a saturated membership function is scaled by its magnitude.
    If True, a saturated input value will return the absolute value as output.
    Otherwise, a saturated input value will always return 1.0.

    Raises
    ------
    `ValueError` - If sizes is given and not None and; the number of names and
    sizes do not match, or the sum of the sizes is not 1.0, or any size is 0.0,
    or any size is less than the overlap.
    """
    if sizes is not None:
        if len(names) != len(sizes):
            raise ValueError(
                "The number of names and sizes must match."
                f"Got; {len(names)=}, {len(sizes)=}."
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
        if any(size < overlap for size in sizes):
            raise ValueError(
                "No size can be less than the overlap."
                f"Got; {sizes=}, {overlap=}."
            )
    # Each membership needs its plateau (of the same size) and its ramps
    # (of the same size). The first and last membership functions only
    # have one ramp (right and left respectively).
    membership_functions: list[TrapezoidalFunction] = []
    if sizes is None:
        spaces = np.linspace(0.0, 1.0, len(names) * 2)
    else:
        sizes = np.cumsum(sizes)[:-1]
        spaces = [0.0]
        for size in sizes:
            spaces.append(size - overlap / 2)
            spaces.append(size + overlap / 2)
        spaces.append(1.0)
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
            TrapezoidalFunction(name, start, first_peak, second_peak, end)
        )
    if include_saturated:
        max_name: str = "max-saturated"
        min_name: str = "min-saturated"
        if isinstance(include_saturated, list):
            max_name = include_saturated[0]
            min_name = include_saturated[1]
        membership_functions.append(
            MaxSaturatedFunction(max_name, scale_saturated)
        )
        membership_functions.append(
            MinSaturatedFunction(min_name, scale_saturated)
        )
    return tuple(membership_functions)


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


class FuzzyControllerTerms(NamedTuple):
    """
    Fuzzy control output terms.

    Items
    -----
    `rule_truths: dict[str, dict[str, float]]` - The truth values of the
    rules' antecedents, maps: rule name -> membership function name -> truth.

    `rule_activations: dict[str, dict[str, float]]` - The activations of the
    rules' consequents, maps: rule name -> membership function name ->
    activation.

    `var_truth_sums: dict[str, float]` - The sum of the truth values of all
    rules for each variable's membership function, maps: variable name -> sum
    of truth values.

    `var_activation_sums: dict[str, float]` - The sum of the activations of
    all rules for each variable's membership function, maps: variable name ->
    sum of activations.

    `var_outputs: dict[str, float]` - The aggregate output of all rules for
    each variable (the activation sum divided by the truth sum if the truth
    sum is not 0.0, otherwise it is 0.0), maps: variable name -> output.
    """

    rule_truths: dict[str, dict[str, float]]
    rule_activations: dict[str, dict[str, float]]
    var_truth_sums: dict[str, float]
    var_activation_sums: dict[str, float]
    var_outputs: dict[str, float]


class FuzzyController(Controller):
    """Class defining a fuzzy controller."""

    __slots__ = {
        "__variables": "Fuzzy variables in the controller.",
        "__mem_funcs": "Membership functions in the controller.",
        "__rules": "Rules in the controller.",
        "__rule_truths": "Truth values of the rules' antecedents.",
        "__rule_activations": "Activations of the rules' consequents.",
        "__var_truth_sums": "Sum of the truth values of each variable's "
                            "membership functions.",
        "__var_activation_sums": "Sum of the activations of each variable's "
                                 "membership functions.",
        "__var_outputs": "Output values of each variable."
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
        """
        Create a fuzzy controller.

        A fuzzy controller is a non-linear system controler that uses a
        sub-set of fuzzy logic to calculate control outputs. The control
        output is calculated by the following steps:

        1. Calculate the truth value of each rule's antecedent.
        2. Calculate the activation of each rule's consequent.
        3. Calculate the sum of the truth values of each variable's
           membership functions.
        4. Calculate the sum of the activations of each variable's
           membership functions.
        5. Calculate the output of each variable (the activation sum divided
           by the truth sum if the truth sum is not 0.0, otherwise it is 0.0).
        6. Calculate the output of the controller (the sum of the outputs of
           each variable).

        Parameters
        ----------
        `variables: Iterable[FuzzyVariable]` - The fuzzy variables in the
        controller. There must be at least one variable.

        `mem_funcs: Iterable[MembershipFunction]` - The membership functions
        in the controller. There must be at least one membership function.

        `rules: Iterable[tuple[str, str, float]] | None` - The rules in the
        controller. Each rule is a tuple of the form (variable name,
        membership function name, output weight). If not given or None, no
        rules are added to the controller.

        For other parameters, see `aloy.control.controllers.Controller`.

        Notes
        -----
        Although a fuzzy controller is a non-linear controller, it can be
        optimised for a given system and setpoint to provide an optimal
        control output simply using proportional variables. However, fuzzy
        controllers also support integral and derivative variables, the use
        of which can improve the robustness of the controller, particularly
        for handling different setpoints, and of physical properties of the
        system being controlled (where the steady state error, rise times, and
        settling times change significantly).
        """
        super().__init__(
            input_limits=input_limits,
            output_limits=output_limits,
            input_trans=input_trans,
            error_trans=error_trans,
            output_trans=output_trans,
            initial_error=initial_error
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

        self.__rule_truths: dict[str, dict[str, float]] = {}
        self.__rule_activations: dict[str, dict[str, float]] = {}
        self.__var_truth_sums: dict[str, float] = {}
        self.__var_activation_sums: dict[str, float] = {}
        self.__var_outputs: dict[str, float] = {}

    def __str__(self) -> str:
        """Return a human-readable string representation of the controller."""
        return f"Fuzzy Controller: total rules = {len(self.__rules)}"

    def __repr__(self) -> str:
        """Return a parseable string representation of the controller."""
        rules: list[tuple[str, str, float]] = [
            (var.name, mem_func.name, output)
            for var, rules in self.__rules.items()
            for mem_func, output in rules.items()
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

    @property
    def variables(self) -> dict[str, FuzzyVariable]:
        """Return the variables in the controller."""
        return self.__variables

    @property
    def mem_funcs(self) -> dict[str, MembershipFunction]:
        """Return the membership functions in the controller."""
        return self.__mem_funcs

    @property
    def rules(self) -> dict[FuzzyVariable, dict[MembershipFunction, float]]:
        """Return the rules in the controller."""
        return self.__rules

    @property
    def terms(self) -> FuzzyControllerTerms:
        """
        Return the fuzzy control terms of the latest control output.

        `FuzzyControllerTerms` - The fuzzy controller terms.
        See `aloy.control.fuzzy.FuzzyControllerTerms` for details.
        """
        return FuzzyControllerTerms(
            self.__rule_truths,
            self.__rule_activations,
            self.__var_truth_sums,
            self.__var_activation_sums,
            self.__var_outputs
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
        """
        Calculate and return the control output.

        The output is the sum of the activation of each rule (the truth of the
        rule multiplied by the output of the rule) divided by the sum of the
        truth of each rule.

        Parameters
        ----------
        `control_input: float` - The control input
        (the measured value of the control variable).

        `setpoint: float` - The control setpoint
        (the desired value of the control variable).

        `delta_time: float` - The time difference since the last call.

        `abs_tol: {float | None} = None` - The absolute tolerance for the
        time difference. If given and not None, if the time difference is
        smaller than the given value, then the integral and derivative
        errors are not updated to avoid precision errors.

        Returns
        -------
        `float` - The control output.
        """
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

        # Calculate the degree of truth and activation of each rule;
        # the weighted average of the activation of each rule.
        rule_truths: dict[str, dict[str, float]] = {
            var.name: {
                mem_func.name: 0.0
                for mem_func in rules
            }
            for var, rules in self.__rules.items()
        }
        rule_activations: dict[str, dict[str, float]] = {
            var.name: {
                mem_func.name: 0.0
                for mem_func in rules
            }
            for var, rules in self.__rules.items()
        }
        for var, rules in self.__rules.items():
            for mem_func, output_weight in rules.items():
                activation: RuleActivation = mem_func.get_activation(
                    var_inputs[var.name],
                    output_weight
                )
                rule_truths[var.name][mem_func.name] = activation.truth
                rule_activations[var.name][mem_func.name] = \
                    activation.activation
        var_truth_sums: dict[str, float] = {
            var.name: sum(rule_truths[var.name].values())
            for var in self.__rules
        }
        var_activation_sums: dict[str, float] = {
            var.name: sum(rule_activations[var.name].values())
            for var in self.__rules
        }

        # Calculate control output (defuzzification); the sum over all
        # variables of the sum of the activation of each rule divided
        # by the sum of the degree of truth of each rule.
        var_outputs: dict[str, float] = {
            var.name: (
                var_activation_sums[var.name] / var_truth_sums[var.name]
                if var_truth_sums[var.name] != 0.0
                else 0.0
            )
            for var in self.__rules
        }
        control_output: float = sum(var_outputs.values())
        control_output = clamp(self.transform_output(control_output),
                               *self.output_limits)

        self.__rule_truths = rule_truths
        self.__rule_activations = rule_activations
        self.__var_truth_sums = var_truth_sums
        self.__var_activation_sums = var_activation_sums
        self.__var_outputs = var_outputs

        self._latest_input = control_input    # type: ignore
        self._latest_error = control_error    # type: ignore
        self._latest_output = control_output  # type: ignore

        return control_output

    def reset(self) -> None:
        """Reset the controller."""
        super().reset()
        for var in self.__variables.values():
            var.reset()


def _main() -> None:
    """Main function."""
    # pylint: disable=import-outside-toplevel
    import matplotlib
    matplotlib.use("TkAgg")
    from aloy.control.controlutils import plot_control

    proportional_var = FuzzyVariable("proportional", -10.0, 10.0, gain=1.0)
    derivative_var = DerivativeFuzzyVariable("derivative", -10.0, 10.0,
                                             gain=0.0)
    integral_var = IntegralFuzzyVariable("integral", -10.0, 10.0, gain=1.0)

    mem_funcs = create_membership_function_set(
        ["tiny", "small", "big", "large"],
        [0.30, 0.20, 0.20, 0.30],
        overlap=0.10,
        include_saturated=True
    )
    for mem_func in mem_funcs:
        print(mem_func.params)
    fig = plot_membership_functions(
        mem_funcs
    )
    fig.show()

    controller = FuzzyController(
        [proportional_var, derivative_var, integral_var],
        mem_funcs
    )
    controller.add_rule("proportional", "large", 1.0)
    controller.add_rule("proportional", "big", 0.5)
    controller.add_rule("proportional", "small", -0.5)
    controller.add_rule("proportional", "tiny", -1.0)
    controller.add_rule("derivative", "large", 1.0)
    controller.add_rule("derivative", "big", 0.5)
    controller.add_rule("derivative", "small", -0.5)
    controller.add_rule("derivative", "tiny", -1.0)
    controller.add_rule("integral", "max-saturated", 2.0)
    controller.add_rule("integral", "large", 1.0)
    controller.add_rule("integral", "big", 0.5)
    controller.add_rule("integral", "small", -0.5)
    controller.add_rule("integral", "tiny", -1.0)

    times = []
    inputs = []
    setpoints = []
    errors = []
    outputs = []

    for i in range(1000):
        output = controller.control_output(1.0, 0.0, 0.1)
        if i % 100 == 0:
            print(f"Control Output [{i}]: {output}")
        if times:
            times.append(times[-1] + 0.1)
        else:
            times.append(0.1)
        inputs.append(controller.latest_input)
        setpoints.append(0.0)
        errors.append(controller.latest_error)
        outputs.append(controller.latest_output)
    controller.reset()

    fig, *_ = plot_control(times, inputs, setpoints, errors, outputs)
    fig.show()
    input()


if __name__ == "__main__":
    _main()
