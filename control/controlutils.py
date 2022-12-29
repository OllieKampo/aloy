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

import time

import numpy as np

from auxiliary.numpyutils import arg_first_where, get_turning_points
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

def simulate_control(control_system: "controllers.ControlledSystem",
                     controller: "controllers.Controller",
                     ticks: int,
                     delta_time: float
                     ) -> float:
    """
    Simulate a control system for a given number of ticks.
    
    Parameters
    ----------
    `control_system : ControlSystem` - The control system to simulate.
    
    `controller : Controller` - The controller to use.
    
    `ticks : int` - The number of ticks to simulate.
    
    `delta_time : float` - The time difference between ticks.
    """
    error_values = np.empty(ticks)
    control_outputs = np.empty(ticks)
    time_points = np.linspace(0.0, delta_time * ticks, ticks)
    
    for tick in range(ticks):
        error = control_system.get_error()
        output = controller.control_output(error, delta_time, abs_tol=None)
        control_system.set_output(output, delta_time)
        error_values[tick] = error
        control_outputs[tick] = output
    
    itae = np.absolute(error_values * time_points)
    points = get_turning_points(error_values)
    if points.size > 0:
        peak_index = points[0]
        itae[peak_index:] *= (points.size + 1)
    
    condition = lambda x: x > 0.0 if error_values[0] < 0.0 else x < 0.0
    rise_index = arg_first_where(condition, error_values, axis=0, invalid_val=-1)
    if rise_index != -1:
        itae[rise_index:] *= (rise_index + 1)
    
    return itae.sum()
