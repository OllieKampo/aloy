############################################################################
############################################################################
## Module defining controllers.                                           ##
##                                                                        ##
## Copyright (C) 2022 Oliver Michael Kamperis                             ##
##                                                                        ##
## This program is free software: you can redistribute it and/or modify   ##
## it under the terms of the GNU General Public License as published by   ##
## the Free Software Foundation, either version 3 of the License, or      ##
## any later version.                                                     ##
##                                                                        ##
## This program is distributed in the hope that it will be useful,        ##
## but WITHOUT ANY WARRANTY; without even the implied warranty of         ##
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the           ##
## GNU General Public License for more details.                           ##
##                                                                        ##
## You should have received a copy of the GNU General Public License      ##
## along with this program. If not, see <https://www.gnu.org/licenses/>.  ##
############################################################################
############################################################################

"""Module defining controllers."""

__copyright__ = "Copyright (C) 2022 Oliver Michael Kamperis"
__license__ = "GPL-3.0"

__all__ = ("Controller",
           "ModularController",
           "ControlledSystem",
           "SystemController",
           "AutoSystemController")

def __dir__() -> tuple[str]:
    """Get the names of the module attributes."""
    return __all__

from abc import abstractmethod, abstractproperty
from contextlib import contextmanager
import inspect
from numbers import Real
import threading
import types
from typing import Callable, Literal, Mapping, Optional, final
import statistics

from control.controlutils import ControllerTimer

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
    def control_output(self, error: float, delta_time: float, abs_tol: float | None = None) -> float:
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

class ControllerCombiner(Controller):
    """
    Class defining controller combiners.
    
    A controller combiner is a controller which combines the output of multiple
    other "inner" controllers. The module controller then combines the
    outputs of the inner controllers to produce a final output.
    """

    __slots__ = {"__inner_controllers" : "The inner controllers.",
                 "__latest_output" : "The latest control output.",
                 "__mode" : "The output combination mode.",
                 "__weights" : "The output combination weights of the inner controllers.",
                 "__weights_update_callback" : "The callback to update the weights."}
    
    __combiner_functions = {"sum" : sum, "mean" : statistics.mean, "median" : statistics.median}

    def __init__(self,
                 controllers: Mapping[str, Controller],
                 mode: Literal["sum", "mean", "median"] = "mean",
                 weights: Mapping[str, float] | None = None,
                 ) -> None:
        """
        Create a controller combiner from a mapping of controllers.
        
        Parameters
        ----------
        `controllers : Mapping[str, Controller]` - The inner controllers.
        Given as a mapping of controller names to controller instances.
        
        `mode : "sum" | "mean" | "median"` - The output combination mode of the inner controller outputs.

        `weights : Mapping[str, float] | None` - The output combination weights of the inner controllers.
        Given as a mapping of controller names to weights.
        If not given or  `None`, then all inner controllers are given equal weight.
        """
        self.__inner_controllers: dict[str, Controller] = dict(controllers)
        self.__latest_output: float = 0.0
        self.__mode: Callable[[list[float]], float] = self.__combiner_functions.get(mode, None)
        if self.__mode is None:
            raise ValueError(f"Invalid mode {mode!r}. Choose from {tuple(self.__combiner_functions)!r}.")
        
        if weights is None:
            value: float = 1.0 / len(self.__inner_controllers)
            self.__weights: dict[str, float] = {name : value for name in self.__inner_controllers}
        else:
            self.__weights: dict[str, float] = tuple(weights)
    
    @property
    def inner_controllers(self) -> dict[str, Controller]:
        """Get the inner controllers."""
        return self.__inner_controllers
    
    @property
    def latest_output(self) -> float:
        """Get the latest control output."""
        return self.__latest_output
    
    @property
    def weights(self) -> dict[str, float]:
        """Get the output combination weights of the inner controllers."""
        return types.MappingProxyType(self.__weights)
    
    def update_weights(self, weights: Mapping[str, float]) -> None:
        """
        Set the output combination weights of the inner controllers.
        
        Parameters
        ----------
        `weights : Mapping[str, float]` - The output combination weights of the inner controllers.
        Given as a mapping of controller names to weights.
        """
        self.__weights.update(weights)
    
    def set_weights_update_callback(self, callback: Callable[[float, float, Mapping[str, float]], Mapping[str, float]] | None) -> None:
        """
        Set a callback to update the output combination weights of the inner controllers.

        The callback is called every time the control output is calculated, with the error to the
        setpoint, latest output, and the current weights as arguments, and must return updated weights.
        If a weight is not present in the returned mapping, then its old value is preserved.

        Parameters
        ----------
        `callback : Callable[[float, float, Mapping[str, float]], Mapping[str, float]]` - The callback to update the weights.
        The signature is: `callback(error, latest_output, weights) -> updated_weights`.
        """
        self.__weights_update_callback = callback

    @property
    def mode(self) -> str:
        """Get the output combination mode."""
        return self.__mode.__name__

    def control_output(self, error: float, delta_time: float, abs_tol: float | None = None) -> float:
        """
        Get the combined control output of the inner controllers.

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
        `float` - The combined control output of the inner controllers.
        """
        if self.__weights_update_callback is not None:
            self.__weights.update(self.__weights_update_callback(error, self.__latest_output, self.__weights))
        outputs: list[float] = [(controller.control_output(error, delta_time, abs_tol) * self.__weights[name])
                                for name, controller in self.__inner_controllers.items()]
        self.__latest_output = self.__mode(outputs)
        return self.__latest_output
    
    def reset(self) -> None:
        """Reset all inner controllers."""
        for controller in self.__inner_controllers.values():
            controller.reset()

class ControlledSystem:
    """
    Mixin for creating classes that are controllable by a system controller.
    
    Must expose an error getting method and a output setting callback method.
    """
    
    @property
    def error_variables(self) -> None | str | tuple[str]:
        """Get the names of the error variables."""
        return None
    
    @abstractmethod
    def get_error(self, var_name: str | None = None) -> float:
        """Get the current error from the setpoint(s) for the control system."""
        raise NotImplementedError
    
    @abstractmethod
    def set_output(self, output: float, delta_time: float, var_name: str | None = None) -> None:
        """Set the control output(s) to the system."""
        raise NotImplementedError

class SystemController:
    """Class defining system controllers."""
    
    __slots__ = {"__controller" : "The underlying controller.",
                 "__system" : "The controlled system.",
                 "__var_name" : "The name of the error variable to get from the controlled system.",
                 "__timer" : "The controller timer used to calculate time differences between ticks.",
                 "__ticks" : "The number of ticks since the last reset."}
    
    def __init__(self,
                 controller: Controller,
                 system: ControlledSystem,
                 error_var_name: str | None = None
                 ) -> None:
        """
        Create a PID system controller.
        
        In contrast to the standard controller, a system controller also
        takes a controlled system as input. The control error is taken from,
        and control output set to, the control system, every time the controller
        is 'ticked', handling time dependent calculations automatically.
        Time differences are calculated by an internal timer.

        Parameters
        ----------
        `controller : Controller` - The controller to use.

        `system : ControlSystem` - The system to control.
        Must implement the `get_error(...)` and `set_output(...)`
        methods of the `ControlledSystem` mixin class.
        
        `error_var_name : {str | None} = None` - The name of the error variable to get from the control system.
        """
        self.__controller: Controller = controller
        self.__system: ControlledSystem = system
        self.__var_name: str | None = error_var_name
        self.__timer: ControllerTimer = ControllerTimer()
        self.__ticks: int = 0
    
    @classmethod
    def from_getsetter(cls,
                       controller: Controller,
                       getter: Callable[[], float],
                       setter: Callable[[float, float], None],
                       ) -> "SystemController":
        """
        Create a system controller from getter and setter callbacks.
        
        This is a convenience method for creating a PID system controller
        from `get_error()` and `set_output()` functions that are not
        attached to a control system.

        Parameters
        ----------
        `controller : Controller` - The controller to use.

        `getter : Callable[[], float]` - The error getter function.
        Must take no arguments and return the error as a float.

        `setter : Callable[[float, float], None]` - The output setter function.
        Must take the control output and the time difference since the last call as arguments.
        """
        system = type("getter_setter_system", (ControlledSystem,), {"get_error" : getter, "set_output" : setter})
        return cls(controller, system)
    
    def __repr__(self) -> str:
        """Get the string representation of the system controller instance."""
        return f"{self.__class__.__name__}(controller={self.__controller!r}, system={self.__system!r}, error_var_name={self.__var_name!r})"
    
    @property
    def system(self) -> ControlledSystem:
        """Get the controlled system."""
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
        output: float = self.__controller.control_output(error, delta_time, abs_tol)
        self.__system.set_output(output, delta_time, self.__var_name)
        return (self.__ticks, error, output, delta_time)
    
    def reset(self) -> None:
        """Reset the controller state."""
        self.__ticks = 0
        self.__timer.reset()
        self.__controller.reset()

@final
class AutoSystemController:
    """
    Class defining an automatic system controller that runs concurrently in a separate thread.

    The contoller can either be ran indefinitely using `run_forever()` until an explicit call to `stop()` is made,
    it can also be ran with loop-like stop conditions using `run_for(iterations, time)` or `run_while(condition)`,
    or it can be ran in a context manager using `context_run_for(iterations, time)` or `context_run_while(condition)`.
    """

    __slots__ = {"__system_controller" : "The underlying system controller.",
                 "__atomic_update_lock" : "Lock for atomic updates in run methods.",
                 "__in_context" : "Whether the controller is currently running in a context.",
                 "__sleep_time" : "The sleep time between control ticks.",
                 "__time_factor" : "The time factor to use when calculating the time difference.",
                 "__data_callback" : "The data callback function.",
                 "__run_forever" : "Whether the controller is currently running indefinitely.",
                 "__condition" : "The condition to check for loop-like stop conditions.",
                 "__thread" : "The thread the controller is running in.",
                 "__lock" : "Lock for atomic updates in the thread.",
                 "__running" : "Whether the controller is currently running.",
                 "__stopped" : "Whether the controller has been stopped."}
    
    def __init__(self,
                 system_controller: SystemController
                 ) -> None:
        """
        Create an automatic system controller.
        
        An automatic system controller encapsulates a normal system controller,
        allowing it to be ran concurrently in a separate thread of control.

        Parameters
        ----------
        `system_controller : SystemController` - The system controller to encapsulate.
        """
        self.__system_controller: SystemController = system_controller
        
        ## Parameters for run methods.
        self.__atomic_update_lock = threading.Lock()
        self.__in_context: bool = False
        self.__sleep_time: float = 0.1
        self.__time_factor: float = 1.0
        self.__run_forever: bool = False
        ## Signature: (iterations, error, output, delta_time, total_time) -> bool
        self.__condition: Callable[[int, float, float, float, float], bool] | None = None
        
        ## Variables for the controller thread.
        self.__thread = threading.Thread(target=self.__run)
        self.__thread.daemon = True
        self.__lock = threading.Lock()
        self.__running = threading.Event()
        self.__stopped = threading.Event()
        ## Signature: (iterations, error, output, delta_time, total_time) -> None
        self.__data_callback: Callable[[int, float, float, float, float], None] = None
        self.__thread.start()
    
    def __repr__(self) -> str:
        """Get the string representation of the automatic system controller instance."""
        return f"{self.__class__.__name__}({self.__system_controller!r})"
    
    @property
    def is_running(self) -> bool:
        """Get whether the controller is running."""
        return self.__running.is_set() or self.__lock.locked()
    
    def __start(self) -> None:
        """Start the controller thread."""
        self.__stopped.clear()
        self.__running.set()
    
    def __stop(self) -> None:
        """Stop the controller thread."""
        self.__running.clear()
        self.__stopped.set()
    
    def __run(self) -> None:
        """Run the controller thread."""
        while True:
            self.__running.wait()
            ## The lock is held whilst the controller is running.
            with self.__lock:
                stop = self.__stopped.is_set()
                error, delta_time, output = self.__system_controller.system.get_error(), 0.0, 0.0
                iterations, total_time = 0, 0.0
                while (not stop
                       and (self.__run_forever
                            or self.__condition(iterations, error, output, delta_time, total_time))):
                    iterations, error, output, delta_time = self.__system_controller.tick(self.__time_factor)
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
            raise RuntimeError("Cannot run an AutoSystemController while it is running in a context manager.")
        if self.is_running:
            raise RuntimeError("Cannot run an AutoSystemController while it is already running.")
    
    def __set_parameters(self,
                         tick_rate: int,
                         time_factor: float,
                         callback: Callable[[int, float, float, float, float], None] | None,
                         run_forever: bool,
                         condition: Callable[[int, float, float, float, float], bool] | None
                         ) -> None:
        """Set the parameters of the controller."""
        if tick_rate <= 0:
            raise ValueError(f"Tick rate must be greater than 0. Got; {tick_rate}.")
        if time_factor <= 0.0:
            raise ValueError(f"Time factor must be greater than 0.0. Got; {time_factor}.")
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
    
    def reset(self, initial_error: float | None = None) -> None:
        """
        Reset the controller state.
        
        If the controller is running, pause execution,
        reset the controller, then resume execution.
        
        See `PIDSystemController.reset()` for further documentation.
        """
        if self.__running.is_set():
            ## Only allow one thread to reset the controller at a time.
            with self.__atomic_update_lock:
                self.__stop()
                ## Wait for the controller to stop before resetting.
                with self.__lock:
                    self.__system_controller.reset(initial_error)
                self.__start()
        else: self.__system_controller.reset(initial_error)
    
    def stop(self) -> None:
        """
        Stop the controller.
        
        Do nothing if the controller is not running.

        Raises
        ------
        `RuntimeError` - If the controller is running in a context manager.
        """
        if self.__in_context:
            raise RuntimeError("Cannot explicitly stop an AutoSystemController while it is running in a context manager.")
        self.__stop()
    
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
        `tick_rate : int = 10` - The tick rate of the controller (ticks per second).
        This is approximate, the actual tick rate may vary, the only
        guarantee is that the tick rate will not exceed the given value.

        `time_factor : float = 1.0` - The time factor to use when calculating time differences.
        The tick rate is multiplied by this value to get the tick rate relative to the time factor.

        `data_callback : Callable[[int, float, float, float, float], None] = None` - A callable to
        callback control data to. Arguments are: `(iteration, error, output, delta time, total time)`.
        
        `reset : bool = True` - Whether to also reset the controller state when the context is exited.

        Raises
        ------
        `RuntimeError` - If the controller is already running.
        """
        try:
            self.run_forever(tick_rate, time_factor, data_callback)
            self.__in_context = True
            yield None
        finally:
            self.__in_context = False
            self.stop()
            if reset:
                self.reset()
    
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
        `tick_rate : int = 10` - The tick rate of the controller (ticks per second).
        This is approximate, the actual tick rate may vary, the only
        guarantee is that the tick rate will not exceed the given value.

        `time_factor : float = 1.0` - The time factor to use when calculating time differences.
        The tick rate is multiplied by this value to get the tick rate relative to the time factor.
        
        `data_callback : Callable[[int, float, float, float, float], None] = None` - A callable to
        callback control data to. Arguments are: `(iteration, error, output, delta time, total time)`.
        
        Raises
        ------
        `RuntimeError` - If the controller is already running.
        """
        with self.__atomic_update_lock:
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
        
        `tick_rate : int = 10` - The tick rate of the controller (ticks per second).
        This is approximate, the actual tick rate may vary, the only
        guarantee is that the tick rate will not exceed the given value.

        `time_factor : float = 1.0` - The time factor to use when calculating time differences.
        The tick rate is multiplied by this value to get the tick rate relative to the time factor.
        
        `data_callback : Callable[[int, float, float, float, float], None] = None` - A callable to
        callback control data to. Arguments are: `(iteration, error, output, delta time, total time)`.
        
        Raises
        ------
        `RuntimeError` - If the controller is already running.
        """
        with self.__atomic_update_lock:
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
        
        `tick_rate : int = 10` - The tick rate of the controller (ticks per second).
        This is approximate, the actual tick rate may vary, the only
        guarantee is that the tick rate will not exceed the given value.

        `time_factor : float = 1.0` - The time factor to use when calculating time differences.
        The tick rate is multiplied by this value to get the tick rate relative to the time factor.
        
        `data_callback : Callable[[int, float, float, float, float], None] = None` - A callable to
        callback control data to. Arguments are: `(iteration, error, output, delta time, total time)`.
        """
        with self.__atomic_update_lock:
            self.__check_running_state()
            self.__set_parameters(tick_rate, time_factor, data_callback, False, condition)
            self.__start()
