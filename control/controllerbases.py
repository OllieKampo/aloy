###########################################################################
###########################################################################
## Base class mixins for controllers.                                    ##
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

"""Module defining base class mixins for controllers."""

__all__ = ("Controller",
           "SystemController",
           "AutoSystemController")

from abc import abstractmethod, abstractproperty
from contextlib import contextmanager
from numbers import Real
from typing import Callable, Optional

class Controller:
    """Base class mixin for creating controller classes."""
    
    __slots__ = ()
    
    @abstractproperty
    def latest_output(self) -> Real:
        """
        Get the latest control output.
        
        `latest_output : float` - The latest output as a float.
        """
        ...
    
    @abstractmethod
    def control_output(self, error: float, delta_time: float, abs_tol: float | None = 1e-6) -> float:
        """
        Get the control output.
        
        Parameters
        ----------
        `error : float` - The error to the setpoint.
        
        `delta_time : float` - The time difference since the last call.
        
        `abs_tol : float` - The absolute tolerance for the time difference.
        If the time difference is smaller than this value, then the integral
        and derivative errors are not updated to avoid precision errors.
        Set to `None` to disable.
        
        Returns
        -------
        `float` - The control output.
        """
        ...
    
    @abstractmethod
    def reset(self) -> None:
        """Reset the controller state."""
        ...

class SystemController(Controller):
    """Base class mixin for creating system controller classes."""
    
    __slots__ = ()
    
    @abstractproperty
    def system(self) -> object:
        """Get the system controlled by this controller."""
        ...
    
    @abstractmethod
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
        ...
    
    @abstractmethod
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
        ...

class AutoSystemController(SystemController):
    """Base class mixin for creating automatic system controller classes."""
    
    __slots__ = ()
    
    @abstractmethod
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
        ...
    
    @abstractmethod
    def tick(self,
             time_factor: float = 1.0,
             abs_tol: float | None = None
             ) -> tuple[int, float, float, float]:
        """
        Manually tick the controller.
        
        It is an error to call this method whilst the controller is running.
        
        See `PIDSystemController.tick()` for further documentation.
        """
        ...
    
    @abstractmethod
    def reset(self, initial_error: float | None = None) -> None:
        """
        Reset the controller state.
        
        If the controller is running, pause execution,
        reset the controller, then resume execution.
        
        See `PIDSystemController.reset()` for further documentation.
        """
        ...
    
    @abstractmethod
    def stop(self, reset: bool = True) -> None:
        """
        Stop the controller.
        
        Do nothing if the controller is not running.
        
        Parameters
        ----------
        `reset : bool` - Whether to also reset the controller state.
        """
        ...
    
    @abstractmethod
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
        ...
    
    @abstractmethod
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
        ...
    
    @abstractmethod
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
        ...
    
    @abstractmethod
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
        ...
