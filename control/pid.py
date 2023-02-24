###########################################################################
###########################################################################
## Proportional-Integral-Derivative (PID) controllers.                   ##
##                                                                       ##
## Copyright (C) 2022 Oliver Michael Kamperis                            ##
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

"""Module defining Proportional-Integral-Derivative (PID) controllers."""

__all__ = ("PIDControllerGains",
           "PIDController")

def __dir__() -> tuple[str]:
    """Get the names of module attributes."""
    return __all__

from collections import deque
from numbers import Real
from typing import Callable, NamedTuple

from control.controllers import Controller, clamp, calc_error

class PIDControllerGains(NamedTuple):
    """
    PID controller gains.
    
    Items
    -----
    `Kp : float` - Proportional gain.
    
    `Ki : float` - Integral gain.
    
    `Kd : float` - Derivative gain.
    """
    
    Kp: float
    Ki: float
    Kd: float

class PIDController(Controller):
    """Class defining PID feedback controllers."""
    
    __slots__ = ("__Kp",
                 "__Ki",
                 "__Kd",
                 "__integral",
                 "__previous_derivatives")
    
    def __init__(self,
                 Kp: float,
                 Ki: float,
                 Kd: float, /,
                 average_derivative: int = 3,
                 input_limits: tuple[float | None, float | None] = (None, None),
                 output_limits: tuple[float | None, float | None] = (None, None),
                 input_transform: Callable[[Real], Real] = lambda x: x,
                 output_transform: Callable[[Real], Real] = lambda x: x,
                 initial_error: float | None = None) -> None:
        """
        Create a PID feedback controller.
        
        A PID controller is a linear system controller whose output is the
        weighted sum of proportional, integral and derivative errors, where;
            - the proportional error is the difference between the desired (setpoint)
              and actual (input) value of the control variable,
            - the integral error is the trapezoidal approximation of the sum of the
              proportional error with respect to time,
            - the derivative error is the (possibly smoothed by moving average) linear
              approximation of the rate of change of the error with respect to time.
        
        The weight applied to each error is called a 'gain', which controls its contribution
        towards the resultant control output. The output is therefore the sum of three terms;
        ```
            error = control_input - setpoint
            output = Kp * error + Ki * integral(error, dt) + Kd * derivative(error, dt)
        ```
        
        The role of the terms are as follows;
            - the propoptional term is the main contribution to the control output towards the setpoint,
            - the integral term exists to eliminate steady-state errors, i.e. errors that require a non-zero
              control output even when the error is very small or zero (and thus the proportional term will
              also be small or zero), to maintain equilibrium. Steady state errors are typically caused by
              some constant force applied to the system that is not dependent on the magnitude of the error,
            - the derivative term damps the rate of change of the error towards the setpoint,
              ensuring a steady approach, minimising overshooting and oscillations.
        
        Parameters
        ----------
        `Kp : float` - The proportional gain, drives approach to setpoint.
        
        `Ki : float` - The integral gain, eliminates steady-state error.
        
        `Kd : float` - The derivative gain, damps approach to setpoint.

        `average_derivative : int = 3` - The number of samples to use for the moving average approximation
        of the derivative error. A value of `1` will use the piecewise linear approximation of the rate
        of change of error between the last consecutive pair of error points (i.e. the gradient between
        the current and previous error). Any value of `n > 1` will use a 'smoothed approximation' by
        taking the moving average over the last `n` gradients of each consecutive pair of points in
        the last `n + 1` error points. 
        
        For other parameters, see `jinx.control.controllers.Controller`.

        Notes
        -----
        Since the integral error accumulates whenever the system's error to the setpoint is non-zero,
        high values for the integral gain relative to the propoptional can cause the integral term to accumlate
        significantly above that needed to eliminate the steady state error, causing overshooting and oscillation.
        
        Damping with the derivative term is most important when the system being controlled is inertial.
        This is because the system may accelerate to a large rate of change in the error, which may be
        difficult to deccelerate before the system reaches the setpoint, potentially causing overshooting.
        """
        super().__init__(input_limits, output_limits,
                         input_transform, output_transform,
                         initial_error)
        ## PID controller gains.
        self.__Kp: float = Kp
        self.__Ki: float = Ki
        self.__Kd: float = Kd
        
        ## PID controller state.
        self.__integral: float = 0.0
        if average_derivative < 1:
            raise ValueError(f"Average derivative window size must be at least 1. Got; {average_derivative}.")
        self.__previous_derivatives: deque[float] = deque(maxlen=average_derivative)
    
    def __str__(self) -> str:
        """Return a human-readable string representation of the PID controller."""
        return f"PID controller with gains: Kp={self.__Kp}, Ki={self.__Ki}, Kd={self.__Kd}."
    
    def __repr__(self) -> str:
        """Return a parseable string representation of the PID controller."""
        return f"PIDcontroller(Kp={self.__Kp}, Ki={self.__Ki}, Kd={self.__Kd}, " \
               f"input_limits={self.__input_limits}, output_limits={self.__output_limits}, " \
               f"initial_error={self.__latest_error}, average_derivative={self.__previous_derivatives.maxlen})"
    
    @property
    def gains(self) -> PIDControllerGains:
        """
        Get or set the PID controller gains.
        
        `(Kp : float, Ki : float, Kd : float)` - The PID controller gains as a 3-tuple.
        """
        return PIDControllerGains(self.__Kp, self.__Ki, self.__Kd)
    
    @gains.setter
    def gains(self, gains: tuple[float, float, float] | PIDControllerGains) -> None:
        """Set the PID controller gains from a 3-tuple."""
        if not len(gains) == 3:
            raise ValueError(f"Expected exactly 3 gains. Got; {len(gains)}.")
        self.__Kp, self.__Ki, self.__Kd = (new_gain if new_gain is not None else original_gain
                                           for new_gain, original_gain in zip(gains, self.gains))
    
    def set_gains(self,
                  Kp: float | None,
                  Ki: float | None,
                  Kd: float | None
                  ) -> None:
        """
        Set the PID controller gains.
        
        Passing a value of None will leave the gain unchanged.
        
        Parameters
        ----------
        `Kp : {float | None}` - The proportional gain.
        
        `Ki : {float | None}` - The integral gain.
        
        `Kd : {float | None}` - The derivative gain.
        """
        if Kp is not None: self.__Kp = Kp
        if Ki is not None: self.__Ki = Ki
        if Kd is not None: self.__Kd = Kd
    
    @property
    def average_derivative(self) -> int:
        """
        Get or set the moving average derivative error window size.
        
        `average_derivative : int` - The number of error points to use for
        the moving average calculation of the derivative error as an integer.
        """
        return self.__previous_derivatives.maxlen
    
    @average_derivative.setter
    def average_derivative(self, average_derivative: int) -> None:
        """Set the moving average derivative error window size."""
        if average_derivative < 1:
            raise ValueError(f"Average derivative window size must be at least 1. Got; {average_derivative}.")
        self.__previous_derivatives = deque(self.__previous_derivatives, maxlen=average_derivative)
    
    def control_output(self,
                       control_input: float,
                       setpoint: float, /,
                       delta_time: float,
                       abs_tol: float | None = None
                       ) -> float:
        """
        Calculate and return the control output.
        
        The output is the sum of the proportional, integral and derivative terms, where;
            - the proportional term is directly proportional to the error itself,
            - the integral term is proportional to the trapzoidal integral of the error over time,
            - the derivative term is proportional to the rate of change of the error over time.
        
        Parameters
        ----------
        `control_input : float` - The control input (the measured value of the control variable).

        `setpoint : float` - The control setpoint (the desired value of the control variable).
        
        `delta_time : float` - The time difference since the last call.
        
        `abs_tol : {float | None} = None` - The absolute tolerance for the time difference.
        If given and not None, then if the time difference is smaller than the given value,
        then the integral and derivative errors are not updated to avoid precision errors.
        
        Returns
        -------
        `float` - The control output.
        """
        if delta_time < 0.0:
            raise ValueError(f"The time difference must be positive. Got; {delta_time}.")
        
        control_input = self.transform_input(control_input)
        error: float = calc_error(control_input, setpoint, *self.input_limits)

        ## Only update the integral and derivative errors if the time difference is large enough.
        derivative: float = 0.0
        if delta_time > 0.0 and (abs_tol is None or delta_time > abs_tol):
            self.__integral += error * delta_time
            if self.latest_error is not None:
                self.__previous_derivatives.append((error - self.latest_error) / delta_time)
                derivative = sum(self.__previous_derivatives) / len(self.__previous_derivatives)
        
        output: float = (self.__Kp * error + self.__Ki * self.__integral + self.__Kd * derivative)
        output = clamp(self.transform_output(output), *self.output_limits)
        
        self._latest_input = control_input
        self._latest_error = error
        self._latest_output = output

        return output
    
    def reset(self) -> None:
        """Reset the controller state."""
        super().reset()
        self.__integral = 0.0
        self.__previous_derivatives.clear()
