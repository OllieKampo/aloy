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
Module defining Proportional-Integral-Derivative (PID) feedback controllers.
"""

from collections import deque
from typing import Callable, NamedTuple

from aloy.control.controllers import Controller, clamp, calc_error

__copyright__ = "Copyright (C) 2023 Oliver Michael Kamperis"
__license__ = "GPL-3.0"
__version__ = "1.0.1"

__all__ = (
    "PIDControllerGains",
    "PIDControllerTerms",
    "PIDController"
)


def __dir__() -> tuple[str, ...]:
    """Get the names of module attributes."""
    return __all__


class PIDControllerGains(NamedTuple):
    """
    PID controller gains.

    Items
    -----
    `pg: float` - Proportional gain.

    `ig: float` - Integral gain.

    `dg: float` - Derivative gain.
    """

    pg: float
    ig: float
    dg: float


class PIDControllerTerms(NamedTuple):
    """
    PID control output terms.

    Items
    -----
    `pt: float` - Proportional output term.

    `it: float` - Integral output term.

    `dt: float` - Derivative output term.
    """

    pt: float
    it: float
    dt: float


class PIDController(Controller):
    """Class defining PID feedback controllers."""

    __slots__ = {
        "__pg": "Proportional gain.",
        "__ig": "Integral gain.",
        "__dg": "Derivative gain.",
        "__pt": "Proportional output term.",
        "__it": "Integral output term.",
        "__dt": "Derivative output term.",
        "__integral": "Integral of the error.",
        "__derivatives": "Previous derivatives of the error."
    }

    def __init__(
        self,
        pg: float,
        ig: float,
        dg: float, /,
        average_derivative: int = 3,
        input_limits: tuple[float | None, float | None] = (None, None),
        output_limits: tuple[float | None, float | None] = (None, None),
        input_trans: Callable[[float], float] | None = None,
        error_trans: Callable[[float], float] | None = None,
        output_trans: Callable[[float], float] | None = None,
        initial_error: float | None = None
    ) -> None:
        """
        Create a PID feedback controller.

        A PID controller is a linear system controller whose output is the
        weighted sum of proportional, integral and derivative errors, where;
            - the proportional error is the difference between the desired
              (setpoint) and actual (input) value of the control variable,
            - the integral error is the trapezoidal approximation of the sum
              of the proportional error with respect to time,
            - the derivative error is the (smoothed by moving average) linear
              approximation of the rate of change of error with time.

        The weight applied to each error is called a 'gain', which controls
        its contribution towards the control output.
        The output is therefore the sum of three terms;
        ```
            error = control_input - setpoint
            output = (
                pg * error
                + ig * integral(error, dt)
                + dg * derivative(error, dt)
            )
        ```

        The role of the terms are as follows;
            - the propoptional term is the main contribution to the control
              output towards the setpoint,
            - the integral term exists to eliminate steady-state errors, i.e.
              errors that require a non-zero control output even when the
              error is very small or zero (and thus the proportional term will
              also be small or zero), to maintain equilibrium. Steady state
              errors are typically caused by some constant force applied to
              the system that is not dependent on the magnitude of the error,
            - the derivative term damps the rate of change of the error
              towards the setpoint, ensuring a steady approach, minimising
              overshooting and oscillations.

        Parameters
        ----------
        `pg: float` - The proportional gain, drives approach to setpoint.

        `ig: float` - The integral gain, eliminates steady-state error.

        `dg: float` - The derivative gain, damps approach to setpoint.

        `average_derivative: int = 3` - The number of samples to use for the
        moving average approximation of the derivative error. A value of `1`
        will use the piecewise linear approximation of the rate of change of
        error between the last consecutive pair of error points (i.e. the
        gradient between the current and previous error). Any value of `n > 1`
        will use a 'smoothed approximation' by taking the moving average over
        the last `n` gradients of each consecutive pair of points in the last
        `n + 1` error points.

        For other parameters, see `aloy.control.controllers.Controller`.

        Notes
        -----
        Since the integral error accumulates whenever the system's error to
        the setpoint is non-zero, high values for the integral gain relative
        to the propoptional can cause the integral term to accumlate
        significantly above that needed to eliminate the steady state error,
        causing overshooting and oscillation.

        Damping with the derivative term is most important when the system
        being controlled is inertial. This is because the system may
        accelerate to a large rate of change in the error, which may be
        difficult to deccelerate before the system reaches the setpoint,
        potentially causing overshooting.
        """
        super().__init__(
            input_limits,
            output_limits,
            input_trans,
            error_trans,
            output_trans,
            initial_error
        )

        # PID controller gains.
        self.__pg: float = pg
        self.__ig: float = ig
        self.__dg: float = dg

        # PID controller output terms.
        self.__pt: float = 0.0
        self.__it: float = 0.0
        self.__dt: float = 0.0

        # PID controller state.
        self.__integral: float = 0.0
        if average_derivative < 1:
            raise ValueError("Average derivative window size must be "
                             f"at least 1. Got; {average_derivative}.")
        self.__derivatives = deque[float](maxlen=average_derivative)

    def __str__(self) -> str:
        """Return a human-readable string representation of the controller."""
        return f"PID controller: gains = {tuple(self.gains)!s}"

    def __repr__(self) -> str:
        """Return a parseable string representation of the PID controller."""
        return (f"PIDcontroller({self.__pg}, {self.__ig}, {self.__dg}, "
                f"average_derivative={self.average_derivative}, "
                f"input_limits={self.input_limits}, "
                f"output_limits={self.output_limits}, "
                f"input_transform={self.input_transform}, "
                f"error_transform={self.input_transform}, "
                f"output_transform={self.output_transform}, "
                f"initial_error={self.initial_error})")

    @property
    def terms(self) -> PIDControllerTerms:
        """
        Get the individual terms of the latest control output.

        `(pt: float, it: float, dt: float)` - The PID control output terms.
        """
        return PIDControllerTerms(self.__pt, self.__it, self.__dt)

    @property
    def gains(self) -> PIDControllerGains:
        """
        Get or set the PID controller gains.

        `(pg: float, ig: float, dg: float)` - The PID controller gains.
        """
        return PIDControllerGains(self.__pg, self.__ig, self.__dg)

    @gains.setter
    def gains(
        self,
        gains: tuple[float, float, float] | PIDControllerGains
    ) -> None:
        """Set the PID controller gains from a 3-tuple."""
        if not len(gains) == 3:
            raise ValueError(f"Expected exactly 3 gains. Got; {len(gains)}.")
        self.set_gains(*gains)

    def set_gains(
        self,
        pg: float | None,
        ig: float | None,
        dg: float | None, /
    ) -> None:
        """
        Set the PID controller gains.

        Passing a value of None will leave the gain unchanged.

        Parameters
        ----------
        `pg: {float | None}` - The proportional gain.

        `ig: {float | None}` - The integral gain.

        `dg: {float | None}` - The derivative gain.
        """
        if pg is not None:
            self.__pg = pg
        if ig is not None:
            self.__ig = ig
        if dg is not None:
            self.__dg = dg

    @property
    def average_derivative(self) -> int:
        """
        Get or set the moving average derivative error window size.

        `average_derivative: int` - The number of error points to use for
        the moving average calculation of the derivative error as an integer.
        """
        return self.__derivatives.maxlen  # type: ignore

    @average_derivative.setter
    def average_derivative(
        self,
        average_derivative: int
    ) -> None:
        """Set the moving average derivative error window size."""
        if average_derivative < 1:
            raise ValueError("Average derivative window size must be "
                             f"at least 1. Got; {average_derivative}.")
        self.__derivatives = deque(self.__derivatives,
                                   maxlen=average_derivative)

    def control_output(
        self,
        control_input: float,
        setpoint: float, /,
        delta_time: float,
        abs_tol: float | None = None
    ) -> float:
        """
        Calculate and return the control output.

        The output is the sum of the proportional,
        integral and derivative terms, where;
            - the proportional term is directly
              proportional to the error itself,
            - the integral term is proportional to the
              trapzoidal integral of the error over time,
            - the derivative term is proportional to the
              rate of change of the error over time.

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

        # Only update the integral and derivative errors if
        # the time difference is sufficiently large enough.
        derivative: float = 0.0
        if delta_time > 0.0 and (abs_tol is None or delta_time > abs_tol):
            self.__integral += control_error * delta_time
            if self.latest_error is not None:
                derivative = (control_error - self.latest_error) / delta_time
                self.__derivatives.append(derivative)
                derivative = sum(self.__derivatives) / len(self.__derivatives)

        # Calculate the PID controller terms.
        self.__pt = self.__pg * control_error
        self.__it = self.__ig * self.__integral
        self.__dt = self.__dg * derivative
        control_output: float = self.__pt + self.__it + self.__dt
        control_output = clamp(self.transform_output(control_output),
                               *self.output_limits)

        self._latest_input = control_input    # type: ignore
        self._latest_error = control_error    # type: ignore
        self._latest_output = control_output  # type: ignore

        return control_output

    def reset(self) -> None:
        """Reset the controller state."""
        super().reset()
        self.__pt = 0.0
        self.__it = 0.0
        self.__dt = 0.0
        self.__integral = 0.0
        self.__derivatives.clear()
