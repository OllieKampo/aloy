###########################################################################
###########################################################################
## Proportional-Integral-Derivative (PID) controllers.                   ##
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

"""Module defining Proportional-Integral-Derivative (PID) controllers."""

__all__ = ("PIDControllerGains",
           "PIDController",
           "PIDSystemController",
           "AutoPIDSystemController")

import inspect
import threading
from collections import deque
from contextlib import contextmanager
from numbers import Real
from typing import Callable, NamedTuple, Optional

from control.controllerbases import (AutoSystemController, Controller,
                                     SystemController)
from control.controlutils import ControllerTimer

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
    """Class defining a basic PID feedback controller."""
    
    __slots__ = ("__Kp",
                 "__Ki",
                 "__Kd",
                 "__initial_error",
                 "__integral",
                 "__previous_error",
                 "__previous_derivatives",
                 "__latest_output")
    
    def __init__(self,
                 Kp: float,
                 Ki: float,
                 Kd: float,
                 initial_error: float | None = None,
                 average_derivative: int = 3
                 ) -> None:
        """
        Create a PID feedback controller.
        
        A PID controller is a linear system controller whose output is the
        weighted sum of proportional, integral and derivative errors, where;
            - the proportional error is the difference between the desired and actual value
              of the control variable,
            - the integral error is the trapezoidal approximation of the sum of the
              proportional error with respect to time,
            - the derivative error is the (possibly smoothed by moving average) linear
              approximation of the rate of change of the error with respect to time.
        
        The weight applied to each error is called a 'gain', this controls its contribution
        towards the resultant control output. The output is therefore the sum of three terms;
        ```
            error = desired_output - actual_output
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
        
        `initial_error : float | None` - The initial value of the error, None if unknown.
        If None, the error value given to the first call to `control_output()` must be used
        as the initial error, and resultantly the integral and derivative terms cannot be
        calculated until the second call (since they require at least two data points).
        
        `average_derivative : int` - The number of samples to use for the moving average approximation
        of the derivative error. A value of `1` will use the piecewise linear approximation of the rate
        of change of error between the last consecutive pair of error points (i.e. the gradient between
        the current and previous error). Any value of `n > 1` will use a 'smoothed approximation' by
        taking the moving average over the last `n` gradients of each consecutive pair of points in
        the last `n + 1` error points. 
        
        Notes
        -----
        Since the integral error accumulates whenever the system's error to the setpoint is non-zero,
        high values for the integral gain relative to the propoptional can cause the integral term to accumlate
        significantly above that needed to eliminate the steady state error, causing overshooting and oscillation.
        
        Damping with the derivative term is most important when the system being controlled is inertial.
        This is because the system may accelerate to a large rate of change in the error, which may be
        difficult to deccelerate before the system reaches the setpoint, potentially causing overshooting.
        
        Also See
        --------
        `jinx.control.systems` - Module containing test control systems for benchmarking controllers.
        
        `jinx.control.examples` - Module containing Jupyter notebook examples for control systems.
        
        `jinx.control.controlutils` - Module containing utility functions for control systems.
        
        `jinx.control.controllerbases.Controller` - Base class for all controllers.
        """
        ## PID controller gains.
        self.__Kp: float = Kp
        self.__Ki: float = Ki
        self.__Kd: float = Kd
        
        ## Keep record of initial error for resetting.
        self.__initial_error: float | None = initial_error
        
        ## PID controller state.
        self.__integral: float = 0.0
        self.__previous_error: float | None = initial_error
        if average_derivative < 1:
            raise ValueError(f"Average derivative window size must be at least 1. Got; {average_derivative}.")
        self.__previous_derivatives: deque[float] = deque(maxlen=average_derivative)
        self.__latest_output: float = 0.0
    
    def __str__(self) -> str:
        """Return a human-readable string representation of the PID controller."""
        return f"PID controller with gains: Kp={self.__Kp}, Ki={self.__Ki}, Kd={self.__Kd}."
    
    def __repr__(self) -> str:
        """Return a parseable string representation of the PID controller."""
        return f"PIDcontroller(Kp={self.__Kp}, Ki={self.__Ki}, Kd={self.__Kd}, " \
               f"initial_error={self.__previous_error}, average_derivative={self.__previous_derivatives.maxlen})"
    
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
    def initial_error(self) -> float | None:
        """
        Get or set the initial error value.
        
        `initial_error : {float | None}` - The initial value of the error, None if unknown.
        """
        return self.__initial_error
    
    @initial_error.setter
    def initial_error(self, initial_error: float | None) -> None:
        """Set the initial error."""
        self.__initial_error = initial_error
    
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
    
    @property
    def latest_output(self) -> float:
        """
        Get the latest control output.
        
        `latest_output : float` - The latest output as a float.
        """
        return self.__latest_output
    
    def control_output(self,
                       error: float,
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
        `error : float` - The error to the setpoint.
        
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
        
        ## Only update the integral and derivative errors if the time difference is large enough.
        derivative: float = 0.0
        if abs_tol is None or abs(delta_time) > abs_tol:
            self.__integral += error * delta_time
            if self.__previous_error is not None:
                self.__previous_derivatives.append((error - self.__previous_error) / delta_time)
                derivative = sum(self.__previous_derivatives) / len(self.__previous_derivatives)
        
        output = (self.__Kp * error + self.__Ki * self.__integral + self.__Kd * derivative)
        self.__previous_error = error
        return output
    
    def reset(self) -> None:
        """Reset the controller state."""
        self.__integral = 0.0
        self.__previous_derivatives.clear()
        self.__latest_output = 0.0

class PIDSystemController(PIDController, SystemController):
    """Class defining a PID feedback system controller."""
    
    __slots__ = ("__system",
                 "__var_name",
                 "__timer",
                 "__ticks")
    
    def __init__(self,
                 system: "ControlSystem",
                 Kp: float,
                 Ki: float,
                 Kd: float,
                 initial_error: float | None = None,
                 average_derivative: int = 3,
                 error_var_name: str | None = None
                 ) -> None:
        """
        Create a PID system controller.
        
        In contrast to the standard PID controller, a system controller also
        takes a control system as input. The control error is taken from, and
        control output set to, the control system, every time the controller
        is 'ticked', handling time dependent calculations automatically.
        
        Parameters
        ----------
        `system : ControlSystem` - The system to control.
        This must be an instance of a class inheriting from ControlSystem,
        and implementing the `get_error()` and `set_output()` method.
        
        `error_var_name : {str | None} = None` - The name of the error variable to get from the control system.
        
        See `PIDController` for the other parameters and further documentation.
        """
        super().__init__(Kp, Ki, Kd, initial_error, average_derivative)
        self.__system: ControlSystem = system
        self.__var_name: str | None = error_var_name
        self.__timer: ControllerTimer = ControllerTimer()
        self.__ticks: int = 0
    
    @classmethod
    def from_getsetter(cls,
                       getter: Callable[[], float],
                       setter: Callable[[float, float], None],
                       Kp: float,
                       Ki: float,
                       Kd: float,
                       initial_error: float | None = None,
                       average_derivative: int = 3
                       ) -> "PIDSystemController":
        """
        Create a PID system controller from getter and setter callbacks.
        
        This is a convenience method for creating a PID system controller
        from `get_error()` and `set_output()` functions that are not
        attached to a control system.
        """
        system = type("getter_setter_system", (ControlSystem,), {"get_error" : getter, "set_output" : setter})
        return cls(system, Kp, Ki, Kd, initial_error, average_derivative)
    
    def __repr__(self) -> str:
        """Get the string representation of the controller instance."""
        return super().__repr__().replace("PIDcontroller(", f"PIDSystemController(system={self.__system}, ", 1)[:-1] + f", var_name={self.__var_name})"
    
    @property
    def system(self) -> "ControlSystem":
        """Get the system controlled by this controller."""
        return self.__system
    
    @property
    def error_var_name(self) -> str | None:
        """Get the name of the error variable controlled by this controller."""
        return self.__var_name
    
    @property
    def ticks(self) -> int:
        """Get the number of times this controller has been ticked since the last reset."""
        return self.__ticks
    
    def time_since_last_ticked(self, time_factor: float = 1.0) -> float:
        """Get the time in seconds since the controller was last ticked."""
        return self.__timer.time_since_last(time_factor)
    
    def control_output(self,
                       error: float | None,
                       delta_time: float | None = None,
                       abs_tol: float | None = None,
                       system_set: bool = True
                       ) -> float:
        """
        Get the control output.
        
        If called with no arguments, this is equivalent to ticking the controller with
        `tick(time_factor=1.0, abs_tol=abs_tol)`, except only the control output is returned.
        
        If instead `error` is given and not None then calculate the output for the given
        error and time difference. If `delta_time` is None, use the time difference since the
        last call, otherwise use the given time difference and reset the time since last tick.
        
        See `PIDController.control_output()` for further documentation.
        
        Parameters
        ----------
        `error : {float | None}` - The error to the setpoint.
        If None, tick the controller and return the control output,
        all other arguments except `abs_tol` are ignored.
        
        `delta_time : {float | None}` - The time difference since the last call.
        If None, use the time difference since the last call, and reset the time since the last tick.
        
        `abs_tol : {float | None} = None` - The absolute tolerance for the time difference, None for no tolerance.
        If the time difference is smaller than this value, then the integral and derivative
        errors are not updated to avoid precision errors.
        
        `system_set : bool = True` - Whether the control output should also be set to the system.
        
        Returns
        -------
        `float` - The control output.
        """
        self.__ticks += 1
        
        if error is None:
            error = self.__system.get_error(self.__var_name)
        
        if delta_time is None:
            delta_time = self.__timer.get_delta_time()
        else: self.__timer.reset_time()
        
        output: float = super().control_output(error, delta_time, abs_tol)
        
        if system_set:
            self.__system.set_output(output, delta_time, self.__var_name)
        
        return output
    
    def tick(self,
             time_factor: float = 1.0,
             abs_tol: float | None = None
             ) -> tuple[int, float, float, float]:
        """
        Tick the controller.
        
        This calculates the current control output and sets it to the system
        through the system's `set_output(output, delta_time)` callback method.
        
        Parameters
        ----------
        `time_factor : float = 1.0` - The time factor to use when calculating the time difference.
        The actual time difference is multiplied by this value, values smaller than 1.0 will therefore
        act as if less time has passed since the last tick, and vice versa for values greater than 1.0.
        
        `abs_tol : float | None = None` - The absolute tolerance for the time difference, None for no tolerance.
        
        Returns
        -------
        `(tick : int, error : float, output : float, delta_time : float)` - The tick number, the error,
        the control output, and the time in seconds since the last tick (multiplied by the time factor).
        """
        self.__ticks += 1
        error: float = self.__system.get_error(self.__var_name)
        delta_time: float = self.__timer.get_delta_time() * time_factor
        output: float = super().control_output(error, delta_time, abs_tol)
        self.__system.set_output(output, delta_time, self.__var_name)
        return (self.__ticks, error, output, delta_time)
    
    def reset(self) -> None:
        """Reset the controller state."""
        self.__ticks = 0
        self.__timer.reset()
        super().reset()

class PIDAutoSystemController(PIDSystemController, AutoSystemController):
    """Class defining an automatic PID system controller that runs concurrently in a separate thread."""
    
    __slots__ = ("__transaction_lock",
                 "__in_context",
                 "__sleep_time",
                 "__time_factor",
                 "__data_callback",
                 "__run_forever",
                 "__condition",
                 "__thread",
                 "__lock",
                 "__running",
                 "__stopped")
    
    def __init__(self,
                 system: "ControlSystem",
                 Kp: float,
                 Ki: float,
                 Kd: float,
                 initial_error: float | None = None,
                 average_derivative: int = 3,
                 error_var_name: str | None = None
                 ) -> None:
        """
        Create an automatic PID system controller.
        
        An automatic system controller can be ran concurrently in a separate thread.
        
        The contoller can either be ran indefinitely using `run_forever()`, until an explicit call to `stop()` is made,
        or it can be ran with loop-like stop conditions using `run_for(iterations, time)` or `run_while(condition)`.
        """
        super().__init__(system, Kp, Ki, Kd, initial_error, average_derivative, error_var_name)
        
        ## Parameters for run methods.
        self.__transaction_lock = threading.Lock()
        self.__in_context: bool = False
        self.__sleep_time: float = 0.1
        self.__time_factor: float = 1.0
        self.__run_forever: bool = False
        ## Signature: (iterations, error, output, delta time, total time) -> bool
        self.__condition: Callable[[int, float, float, float, float], bool] | None = None
        
        ## Variables for the controller thread.
        self.__thread = threading.Thread(target=self.__run)
        self.__thread.daemon = True
        self.__lock = threading.Lock()
        self.__running = threading.Event()
        self.__stopped = threading.Event()
        ## Signature: (iterations, error, output, delta time, total time) -> None
        self.__data_callback: Callable[[int, float, float, float, float], None] = None
        self.__thread.start()
    
    def __repr__(self) -> str:
        """Get the string representation of the controller instance."""
        return "Auto" + super().__repr__()
    
    @property
    def is_running(self) -> bool:
        """Get whether the controller is running."""
        return self.__running.is_set() or self.__lock.locked()
    
    def __start(self) -> None:
        """Start the controller thread."""
        self.__running.set()
        self.__stopped.clear()
    
    def __stop(self) -> None:
        """Stop the controller thread."""
        self.__stopped.set()
        self.__running.clear()
    
    def __run(self) -> None:
        """Run the controller thread."""
        while True:
            self.__running.wait()
            ## The lock is held whilst the controller is running.
            with self.__lock:
                stop = self.__stopped.is_set()
                error, delta_time, output = self.system.get_error(), 0.0, 0.0
                iterations, total_time = 0, 0.0
                while (not stop
                       and (self.__run_forever
                            or self.__condition(iterations, error, output, delta_time, total_time))):
                    iterations, error, output, delta_time = super().tick(self.__time_factor)
                    total_time += delta_time
                    self.__data_callback(iterations, error, output, delta_time, total_time)
                    ## Preempt stop calls.
                    stop = self.__stopped.wait(self.__sleep_time)
            if not stop: self.__stop()
    
    def __check_running_state(self) -> None:
        """
        Check the running state of the controller.
        
        Raise an error if the controller is running.
        """
        if self.__in_context:
            raise RuntimeError("Cannot run an PIDAutoSystemController while it is running in a context manager.")
        if self.is_running:
            raise RuntimeError("Cannot run an PIDAutoSystemController while it is already running.")
    
    def __set_parameters(self,
                         tick_rate: int,
                         time_factor: float,
                         callback: Callable[[int, float, float, float, float], None] | None,
                         run_forever: bool,
                         condition: Callable[[int, float, float, float, float], bool] | None) -> None:
        """Set the parameters of the controller."""
        self.__sleep_time = (1.0 / tick_rate) / time_factor
        self.__time_factor = time_factor
        if callback is not None:
            if not callable(callback):
                raise TypeError("The data callback must be a callable.")
            if not (num_params := len(inspect.signature(callback).parameters)) == 5:
                raise TypeError(f"The data callback must take five arguments. Given callback takes {num_params}.")
        self.__data_callback = callback
        self.__run_forever = run_forever
        if condition is not None:
            if not callable(condition):
                raise TypeError("The condition must be a callable.")
            if not (num_params := len(inspect.signature(condition).parameters)) == 5:
                raise TypeError(f"The condition must take five arguments. Given callable takes {num_params}.")
        self.__condition = condition
    
    def control_output(self,
                       error: float | None = None,
                       delta_time: float | None = None,
                       abs_tol: float | None = None,
                       system_set: bool = True
                       ) -> float:
        """
        Get the control output.
        
        It is an error to call this method whilst the controller is running.
        
        See `PIDSystemController.control_output()` for further documentation.
        """
        if self.is_running:
            raise RuntimeError("Cannot manually calculate control output from a controller that is running.")
        return super().control_output(error, delta_time, abs_tol, system_set)
    
    def tick(self,
             time_factor: float = 1.0,
             abs_tol: float | None = None
             ) -> tuple[int, float, float, float]:
        """
        Manually tick the controller.
        
        It is an error to call this method whilst the controller is running.
        
        See `PIDSystemController.tick()` for further documentation.
        """
        if self.is_running:
            raise RuntimeError("Cannot manually tick a controller that is running.")
        return super().tick(time_factor, abs_tol)
    
    def reset(self, initial_error: float | None = None) -> None:
        """
        Reset the controller state.
        
        If the controller is running, pause execution,
        reset the controller, then resume execution.
        
        See `PIDSystemController.reset()` for further documentation.
        """
        if self.__running.is_set():
            ## Only allow one thread to reset the controller at a time.
            with self.__transaction_lock:
                self.__stop()
                ## Wait for the controller to stop before resetting.
                with self.__lock:
                    super().reset(initial_error)
                self.__start()
        else: super().reset(initial_error)
    
    def stop(self, reset: bool = True) -> None:
        """
        Stop the controller.
        
        Do nothing if the controller is not running.
        
        Parameters
        ----------
        `reset : bool` - Whether to also reset the controller state.
        """
        if self.__in_context:
            raise RuntimeError("Cannot explicitly stop an PIDAutoSystemController while it is running in a context manager.")
        self.__stop()
        if reset:
            ## Wait for the controller to stop before resetting.
            with self.__lock:
                super().reset()
    
    @contextmanager
    def context_run(self,
                    tick_rate: int = 10,
                    time_factor: float = 1.0,
                    data_callback: Callable[[int, float, float, float, float], None] | None = None,
                    reset: bool = True
                    ) -> None:
        """
        Start the controller in a with-statement context.
        
        The controller is stopped automatically when the context is exited.
        
        For example:
        ```
        with controller.start(tick_rate, time_factor, reset):
            ## Do stuff concurrently...
        ```
        Is equivalent to:
        ```
        try:
            controller.run_forever(tick_rate, time_factor)
            ## Do stuff concurrently...
        finally:
            controller.stop(reset)
        ```
        
        Parameters
        ----------
        `tick_rate : int` - The number of ticks per second.
        This is approximate, the actual tick rate may vary, the only
        guarantee is that the tick rate will not exceed the given value.
        
        `time_factor : float` - The time factor to use when calculating the delta time.
        
        `data_callback : Callable[[int, float, float, float, float], None]` - A callable to
        callback control data to. Arguments are: `(iteration, error, output, delta time, total time)`.
        
        `reset : bool` - Whether to also reset the controller state when the context is exited.
        """
        try:
            self.run_forever(tick_rate, time_factor, data_callback)
            self.__in_context = True
            yield None
        finally:
            self.__in_context = False
            self.stop(reset)
    
    def run_forever(self,
                    tick_rate: int = 10,
                    time_factor: float = 1.0,
                    data_callback: Callable[[int, float, float, float, float], None] = None
                    ) -> None:
        """
        Run the controller in a seperate thread until a stop call is made.
        
        This is a non-blocking call.
        
        Parameters
        ----------
        `tick_rate : int` - The number of ticks per second.
        This is approximate, the actual tick rate may vary, the only
        guarantee is that the tick rate will not exceed the given value.
        
        `time_factor : float` - The time factor to use when calculating the delta time.
        
        `data_callback : Callable[[int, float, float, float, float], None]` - A callable to
        callback control data to. Arguments are: `(iteration, error, output, delta time, total time)`.
        
        Raises
        ------
        `RuntimeError` - If the controller is already running.
        """
        with self.__transaction_lock:
            self.__check_running_state()
            self.__set_parameters(tick_rate, time_factor, data_callback, True, None)
            self.__start()
    
    def run_for(self,
                max_ticks: Optional[int] = None,
                max_time: Optional[Real] = None,
                tick_rate: int = 10,
                time_factor: float = 1.0,
                data_callback: Callable[[int, float, float, float, float], None] = None
                ) -> None:
        """
        Run the controller for a given number of ticks or amount of time (in seconds).
        
        The controller stops when either the tick or time limit is reached.
        
        Parameters
        ----------
        `max_ticks : int` - The maximum number of ticks to run for.
        
        `max_time : float` - The maximum amount of seconds to run for.
        
        `tick_rate : int` - The number of ticks per second.
        This is approximate, the actual tick rate may vary, the only
        guarantee is that the tick rate will not exceed the given value.
        
        `time_factor : float` - The time factor to use when calculating the delta time.
        
        `data_callback : Callable[[int, float, float, float, float], None]` - A callable to
        callback control data to. Arguments are: `(iteration, error, output, delta time, total time)`.
        
        Raises
        ------
        `RuntimeError` - If the controller is already running.
        """
        with self.__transaction_lock:
            self.__check_running_state()
            condition = None
            if max_ticks is not None and max_time is not None:
                condition = lambda i, e, o, dt, tt: i < max_ticks and tt < max_time
            elif max_ticks is not None:
                condition = lambda i, e, o, dt, tt: i < max_ticks
            elif max_time is not None:
                condition = lambda i, e, o, dt, tt: tt < max_time
            self.__set_parameters(tick_rate, time_factor, data_callback, condition is None, condition)
            self.__start()
    
    def run_while(self,
                  condition: Callable[[int, float, float, float, float], bool],
                  tick_rate: int = 10,
                  time_factor: float = 1.0,
                  data_callback: Callable[[int, float, float, float, float], None] = None
                  ) -> None:
        """
        Run the controller while a condition is true.
        
        Parameters
        ----------
        `condition : Callable[[int, float, float, float], bool]` - The condition to test.
        The function signature is `(iterations, error, output, delta time, total time) -> bool`,
        the controller will stop when the condition returns `False`.
        
        `tick_rate : int` - The number of ticks per second.
        This is approximate, the actual tick rate may vary, the only
        guarantee is that the tick rate will not exceed the given value.
        
        `time_factor : float` - The time factor to use when calculating the delta time.
        
        `data_callback : Callable[[int, float, float, float, float], None]` - A callable to
        callback control data to. Arguments are: `(iteration, error, output, delta time, total time)`.
        """
        with self.__transaction_lock:
            self.__check_running_state()
            self.__set_parameters(tick_rate, time_factor, data_callback, False, condition)
            self.__start()

from control.systems import ControlSystem
