###########################################################################
###########################################################################
## Module defining utility classes and functions for using controllers.  ##
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

"""Module defining utility classes and functions for using controllers."""

__all__ = ("ControllerTimer",
           "simulate_control")

from collections import deque
import time
from typing import Callable, Literal

import numpy as np
import matplotlib.pyplot as plt

from auxiliary.numpyutils import get_turning_points
import control.controllers as controllers

class ControllerTimer:
    """Class definng controller timers."""
    
    __slots__ = ("__time_last",
                 "__calls")
    
    def __init__(self):
        """
        Create a new controller timer.
        
        The timer does not start tracking time until the first call to `get_delta_time()`.
        """
        self.__time_last: float | None = None
        self.__calls: int = 0
    
    @property
    def calls(self) -> int:
        """Get the number of delta time calls since the last reset."""
        return self.__calls
    
    def get_delta_time(self, time_factor: float = 1.0) -> float:
        """
        Get the time difference since the last call to this method.
        
        If this is the first call since the timer was created or reset then return `0.0`.
        """
        self.__calls += 1
        time_now = time.perf_counter()
        if self.__time_last is None:
            self.__time_last = time_now
            return 0.0
        raw_time = (time_now - self.__time_last)
        self.__time_last = time_now
        return raw_time * time_factor
    
    def time_since_last(self, time_factor: float = 1.0) -> float:
        """Get the time since the last call to `get_delta_time()` without updating the timer."""
        if self.__time_last is None:
            return 0.0
        return (time.perf_counter() - self.__time_last) * time_factor
    
    def reset(self) -> None:
        """Reset the timer to a state as if it were just created."""
        self.__time_last = None
        self.__calls = 0
    
    def reset_time(self) -> None:
        """
        Reset the timer's internal time to the current time.
        
        The next call to `get_delta_time()` will be calculated with respect to the time this method was called.
        """
        self.__time_last = time.perf_counter()

class WeightUpdateFunctionBuilder:
    """
    Class defining a builder for weight update functions.
    
    Instances of this class are used to create weight update functions.
    A single weight function is used for a controller combiner,
    where each function name is the name of a controller.
    A seperate weight function is needed for each control output of a modular controller,
    where each function name is the name of a control input that maps to that output.
    """

    __slots__ = ("__functions",)

    def __init__(self) -> None:
        """Create a new weight update function builder."""
        # Function signature: (error: float, output: float, current_weight: float) -> new_weight: float
        self.__functions: dict[str, Callable[[float, float, float], float]] = {}
    
    def add_ramp(self, name: str, start_weight: float, end_weight: float) -> None:
        pass
    
    def add_step(self, name: str, weight_left: float, weight_right: float, error: float) -> None:
        pass
    
    def add_sine(self, name: str, start_weight: float, end_weight: float) -> None:
        pass

    def add_slider(self, *names: str, start_weight: float, end_weight: float, type_: Literal["ramp", "step", "sine"]) -> None:
        pass

    def build(self) -> dict[str, Callable[[float, float, float], float]]:
        """Build the weight update functions."""
        return self.__functions

def simulate_control(control_system: "controllers.ControlledSystem",
                     controller: "controllers.Controller",
                     ticks: int,
                     delta_time: float,
                     penalise_oscillation: bool = True,
                     penalise_overshoot: bool = True,
                     lead_ticks: int = 1,
                     lag_ticks: int = 1
                     ) -> float:
    """
    Simulate a control system for a given number of ticks.
    
    Parameters
    ----------
    `control_system : ControlledSystem` - The control system to simulate.
    
    `controller : Controller` - The controller to use.
    
    `ticks : int` - The number of ticks to simulate.
    
    `delta_time : float` - The time difference between ticks.
    
    `penalise_oscillation : bool` - Whether to penalise oscillation.

    `penalise_overshoot : bool` - Whether to penalise overshoot.

    `lead_ticks : int` - The number of ticks to use for lead simulation.

    `lag_ticks : int` - The number of ticks to use for lag simulation.
    """
    if ticks < 1:
        raise ValueError(f"Ticks must be greater than or equal to 1. Got; {ticks}.")
    if delta_time <= 0.0:
        raise ValueError(f"Delta time must be greater than 0. Got; {delta_time}.")
    if lead_ticks < 1:
        raise ValueError(f"Lead ticks must be greater than or equal to 1. Got; {lead_ticks}.")
    if lag_ticks < 1:
        raise ValueError(f"Lag ticks must be greater than or equal to 1. Got; {lag_ticks}.")
    
    setpoint: float = control_system.get_setpoint()
    error_values = np.empty(ticks)
    control_outputs = np.empty(ticks)

    if lead_ticks == 1 and lag_ticks == 1:
        for tick in range(ticks):
            control_input = control_system.get_control_input()
            control_output = controller.control_output(control_input, setpoint, delta_time, abs_tol=None)
            control_system.set_control_output(control_output, delta_time)
            error_values[tick] = controller.latest_error
            control_outputs[tick] = control_output
    else:
        input_queue = deque(maxlen=lead_ticks)
        output_queue = deque(maxlen=lag_ticks)
        for tick in range(ticks):
            control_input = control_system.get_control_input()
            input_queue.append(control_input)
            if len(input_queue) == lead_ticks:
                control_input = input_queue.popleft()
                control_output = controller.control_output(control_input, setpoint, delta_time, abs_tol=None)
                output_queue.append(control_output)
                if len(output_queue) == lag_ticks:
                    control_output = output_queue.popleft()
                    control_system.set_control_output(control_output, delta_time)
            error_values[tick] = controller.latest_error
            control_outputs[tick] = control_output
    
    ## Calculate integral of absolute error over time.
    itae = np.absolute(error_values * delta_time)

    ## Penalise osscillation by multiplying the error between turning points,
    ## where later turning points are penalised more than earlier ones.
    if penalise_oscillation:
        points = get_turning_points(error_values)
        if points.size > 0:
            for i in range(points.size - 1):
                itae[points[i]:points[i + 1]] *= ((points.size + 1) - i)
            itae[points[-1]:] *= 2.0
    
    ## Penalise overshoot by doubling the overshooting error values.
    if penalise_overshoot:
        if error_values[0] < 0.0:
            points = error_values > 0.0
        else:
            points = error_values < 0.0
        itae[points] *= 2.0
    
    return itae.sum()

def plot_error(time_points: np.ndarray,
               error_values: np.ndarray,
               output_values: np.ndarray,
               title_space: float = 0.09,
               plot_gap: float = 0.175,
               width: float = 8,
               height: float = 4
               ) -> None:
    """Plot the control error and output over time with matplotlib."""
    figure, (error_axis, output_axis) = plt.subplots(1, 2)
    figure.suptitle("Control error and Output over Time")
    error_axis.plot(time_points, error_values, color="red", label="Error")
    error_axis.set_xlabel("Time"); error_axis.set_ylabel("Error")
    output_axis.plot(time_points, output_values, color="cyan", label="Output")
    output_axis.set_xlabel("Time"); output_axis.set_ylabel("Output")
    figure.tight_layout()
    figure.subplots_adjust(top=(1.0 - title_space), wspace=plot_gap)
    figure.set_size_inches(width, height)
