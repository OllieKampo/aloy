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

from collections import deque
import inspect
from numbers import Real
import threading
import time
from typing import Callable, Optional

__all__ = ("PIDcontroller",
           "PIDSystemController",
           "AutoPIDSystemController")

class PIDController:
    """Class defining a basic PID feedback controller."""
    
    __slots__ = ("__Kp",
                 "__Ki",
                 "__Kd",
                 "__integral",
                 "__previous_error",
                 "__derivatives")
    
    def __init__(self, Kp: float, Ki: float, Kd: float, initial_error: float | None = None, average_derivative: int = 3) -> None:
        """
        Create a PID feedback controller.
        
        A PID controller is a linear system controller whose output is
        the sum of proportional, integral and derivative error terms.
        
        Each error term is weighted by a gain, such that the output is;
        ```
        error = desired_output - actual_output
        output = Kp * error + Ki * integral(error, dt) + Kd * derivative(error, dt)
        ```
        
        Parameters
        ----------
        `Kp : float` - The proportional gain.
        The propoptional term is the main contribution to the control output towards the setpoint.
        
        `Ki : float` - The integral gain.
        The integral term exists to deal with steady-state errors, i.e. errors that require a large
        control output even when the error is very small or zero (and thus the proportional term will
        also be small or zero), to maintain equilibrium.
        
        `Kd : float` - The derivative gain.
        The derivative term is used to handle with the rate of change of the error,
        ensuring a steady approach towards the setpoint to minimise overshooting and oscillations.
        
        `initial_error : float | None` - The initial value of the error, None if unknown.
        
        `average_derivative : int` - The number of samples to use for the moving average calculation of the derivative error.
        
        Notes
        -----
        Intuitively, large proportional gain (Kp) will greater control output relative
        to the error and more rapid response to changes in the error.
        
        Large derivative gain (Kd) will damp the control output towards the setpoint.
        This is especially useful for systems with a slow response time and high inertia,
        because the system may accelerate to a large rate of change in the error (particularly
        due to accumulation of the integral term) that will be difficult to deccelerate before
        the system reaches the setpoint and cause overshooting unless damping is used.
        
        Low values for the integral gain usually causes a steady increase in the control output,
        and thus a gradual approach towards the setpoint that is likely to avoid overshooting.
        
        High values for the integral gain relative to the propoptional usually causes a significant accumulation in the integral error
        (and thus the control output) whilst the system's error to the setpoint is non-zero.
        This can cause the integral error to accumlate above that needed to eliminate the steady state error,
        cuasing overshooting and oscillation.
        """
        ## PID controller gains.
        self.__Kp: float = Kp
        self.__Ki: float = Ki
        self.__Kd: float = Kd
        
        ## PID controller state.
        self.__previous_error: Optional[float] = initial_error
        self.__integral: float = 0.0
        self.__derivatives: deque[float] = deque(maxlen=average_derivative)
    
    def __str__(self) -> str:
        return f"PID controller with gains Kp={self.Kp}, Ki={self.Ki}, Kd={self.Kd}"
    
    def __repr__(self) -> str:
        return f"PIDcontroller(Kp={self.Kp}, Ki={self.Ki}, Kd={self.Kd})"
    
    def set_gains(self, Kp: float, Ki: float, Kd: float) -> None:
        """Set the PID controller gains."""
        self.__Kp = Kp
        self.__Ki = Ki
        self.__Kd = Kd
    
    def get_output(self, error: float, delta_time: float, abs_tol: float | None = 1e-6) -> float:
        """
        Get the control output.
        
        The output is the sum of the proportional, integral and derivative terms, where;
            - the proportional term is directly proportional to the error itself,
            - the integral term is proportional to the sum of the average error over time (trapzoidal integration),
            - the derivative term is proportional to the rate of change of the error.
        
        Parameters
        ----------
        `error : float` - The error to the setpoint.
        
        `delta_time : float` - The time difference since the last call.
        
        `abs_tol : float` - The absolute tolerance for the time difference.
        If the time difference is smaller than this value, then the integral
        and derivative errors are not updated to avoid precision errors.
        Set to `None` to disable.
        """
        if delta_time < 0.0:
            raise ValueError(f"The time difference must be positive. Got {delta_time}.")
        
        ## Only update the integral and derivative errors if the time difference is large enough.
        derivative: float = 0.0
        if abs_tol is None or abs(delta_time) < abs_tol:
            self.__integral += error * delta_time
            if self.__previous_error is not None:
                derivative = (error - self.__previous_error) / delta_time
                self.__derivatives.append(derivative)
                derivative = sum(self.__derivatives) / len(self.__derivatives)
        
        output = (self.__Kp * error + self.__Ki * self.__integral + self.__Kd * derivative)
        self.__previous_error = error
        return output
    
    def reset(self) -> None:
        """Reset the PID controller state."""
        self.__integral = 0.0
        self.__previous_error = 0.0

class PIDSystemController(PIDController):
    """Class defining a PID feedback system controller."""
    
    __slots__ = ("__system",
                 "__time_last",
                 "__latest_output")
    
    def __init__(self, Kp: float, Ki: float, Kd: float, system: "ControlSystem") -> None:
        """
        Create a PID system controller.
        
        In contrast to the standard PID controller, a system controller also
        takes a control system as input. The control error is taken from, and
        control output set to, the control system, every time the controller
        is 'ticked', handling time dependent calculations automatically.
        """
        super().__init__(Kp, Ki, Kd)
        self.__system: ControlSystem = system
        self.__time_last: Optional[int | float] = None
        self.__latest_output: float = 0.0
    
    @property
    def system(self) -> "ControlSystem":
        """Get the system controlled by this controller."""
        return self.__system
    
    def __get_delta_time(self, time_factor: float) -> float | int:
        """Get the time difference since the last call."""
        time_now = time.perf_counter()
        if self.__time_last is None:
            self.__time_last = time_now
            return 0.0
        else:
            raw_time = (time_now - self.__time_last)
            self.__time_last = time_now
            return raw_time * time_factor
    
    def get_output(self, error: float = 0.0, delta_time: float = 0.0, abs_tol: float | None = 0.000001) -> float:
        """Get the control output from the most recent tick, arguments are ignored."""
        return self.__latest_output
    
    def tick(self, time_factor: float = 1.0) -> tuple[float, float, float]:
        """
        Tick the controller.
        
        This calculates the current control output and sets it to the system
        throguh the system's `set_output(output, delta_time)` callback method.
        
        Returns
        -------
        `(error : float, delta_time : float, output : float)` - The error, time
        since the last tick (in seconds) and the control output.
        """
        error = self.__system.get_error()
        delta_time = self.__get_delta_time(time_factor)
        self.__latest_output = super().get_output(error, delta_time)
        self.__system.set_output(self.__lastest_output, delta_time)
        return (error, delta_time, self.__latest_output)
    
    def reset(self) -> None:
        """Reset the controller state."""
        self.__time_last = None
        super().reset()

class AutoPIDSystemController(PIDSystemController):
    """Class defining an automatic PID system controller that runs concurrently in a separate thread."""
    
    __slots__ = ("__thread",
                 "__lock",
                 "__running",
                 "__stopped",
                 "__sleep_time",
                 "__time_factor",
                 "__run_forever",
                 "__condition")
    
    def __init__(self, Kp: float, Ki: float, Kd: float, system: "ControlSystem") -> None:
        """
        Create an automatic PID system controller.
        
        An automatic system controller can be ran concurrently in a separate thread.
        
        The contoller can either be ran indefinitely using `run()`, until an explicit call to `stop()` is made,
        or it can be ran with loop-like stop conditions using `run_for(iterations, time)` or `run_while(condition)`.
        """
        super().__init__(Kp, Ki, Kd, system)
        
        ## Parameters for run methods.
        self.__sleep_time: float = 0.1
        self.__time_factor: float = 1.0
        self.__run_forever: bool = False
        self.__condition: Optional[Callable[[int, float, float, float], bool]] = None
        
        ## Variables for the controller thread.
        self.__thread = threading.Thread(target=self.__run)
        self.__thread.daemon = True
        self.__lock = threading.Lock()
        self.__running = threading.Event()
        self.__stopped = threading.Event()
        self.__thread.start()
    
    @property
    def is_running(self) -> bool:
        """Get whether the controller is running."""
        return self.__running.is_set()
    
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
                stop = False
                error, delta_time, output = self.system.get_error(), 0.0, 0.0
                iterations, total_time = 0, 0.0
                while (not stop
                       and (self.__run_forever
                            or self.__condition(iterations, error, total_time, output))):
                    error, delta_time, output = super().tick(self.__time_factor)
                    iterations += 1
                    total_time += delta_time
                    ## Preempt stop calls.
                    stop = self.__stopped.wait(self.__sleep_time)
            if not stop: self.__stop()
    
    def tick(self, time_factor: float = 1) -> None:
        """
        Manually tick the controller.
        
        It is an error to tick a controller that is running.
        """
        if self.__running.is_set():
            raise RuntimeError("Cannot tick an AutoPIDSystemController while it is running.")
        return super().tick(time_factor)
    
    def reset(self) -> None:
        """
        Reset the controller state.
        
        If the controller is running, pause execution,
        reset the controller, then resume execution.
        """
        if self.__running.is_set():
            self.__stop()
            ## Wait for the controller to stop before resetting.
            with self.__lock:
                super().reset()
            self.__start()
        else: super().reset()
    
    def stop(self, reset: bool = True) -> None:
        """
        Stop the controller.
        
        Do nothing if the controller is not running.
        """
        self.__stop()
        if reset:
            ## Wait for the controller to stop before resetting.
            with self.__lock:
                super().reset()
    
    def run(self,
            tick_rate: int = 10,
            time_factor: float = 1.0
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
        
        Raises
        ------
        `RuntimeError` - If the controller is already running.
        """
        if self.__running.is_set():
            raise RuntimeError("Cannot run an AutoPIDSystemController while it is already running.")
        self.__sleep_time = 1.0 / tick_rate
        self.__time_factor = time_factor
        self.__run_forever = True
        self.__condition = None
        self.__running.set()
    
    def run_for(self,
                max_ticks: Optional[int] = None,
                max_time: Optional[Real] = None,
                tick_rate: int = 10,
                time_factor: float = 1.0
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
        
        Raises
        ------
        `RuntimeError` - If the controller is already running.
        """
        if self.__running.is_set():
            raise RuntimeError("Cannot run an AutoPIDSystemController while it is already running.")
        self.__sleep_time = 1.0 / tick_rate
        self.__time_factor = time_factor
        self.__run_forever = False
        if max_ticks is not None and max_time is not None:
            self.__condition = lambda i, e, t, o: i < max_ticks and t < max_time
        elif max_ticks is not None:
            self.__condition = lambda i, e, t, o: i < max_ticks
        elif max_time is not None:
            self.__condition = lambda i, e, t, o: t < max_time
        else: self.__run_forever = True
        self.__running.set()
    
    def run_while(self,
                  condition: Callable[[int, float, float, float], bool],
                  tick_rate: int = 10,
                  time_factor: float = 1.0
                  ) -> None:
        """
        Run the controller while a condition is true.
        
        Parameters
        ----------
        `condition : Callable[[int, float, float, float], bool]` - The condition to test.
        The function signature is `(iterations, error, total_time, output) -> bool`,
        the controller will stop when the condition returns `False`.
        
        `tick_rate : int` - The number of ticks per second.
        This is approximate, the actual tick rate may vary, the only
        guarantee is that the tick rate will not exceed the given value.
        
        `time_factor : float` - The time factor to use when calculating the delta time.
        """
        if self.__running.is_set():
            raise RuntimeError("Cannot run an AutoPIDSystemController while it is already running.")
        self.__sleep_time = 1.0 / tick_rate
        self.__time_factor = time_factor
        self.__run_forever = False
        if not callable(condition):
            raise TypeError("The condition must be a callable.")
        if not len(inspect.signature(condition).parameters) == 4:
            raise TypeError("The condition must take four arguments.")
        self.__condition = condition
        self.__running.set()

from control.systems import ControlSystem