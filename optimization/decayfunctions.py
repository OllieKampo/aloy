###########################################################################
###########################################################################
## Module defining parameter decay functions for optimisation algorithms.##
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

"""Module defining parameter decay functions for optimisation algorithms."""

__all__ = ("DecayFunctionType",
           "linear_decay_function",
           "polynomial_decay_function",
           "exponential_decay_function",
           "sinusoidal_decay_function",
           "get_decay_function")

import math
from typing import Callable, Literal, TypeAlias

DecayFunctionType: TypeAlias = Literal["lin", "pol", "exp", "b-exp", "sin"]

def linear_decay_function(initial_value: float,
                          final_value: float,
                          decay_start: int,
                          decay_end: int,
                          decay_rate: float = 1.0
                          ) -> Callable[[int], float]:
    """
    Create a linear parameter decay function.
    
    The parameter value reduces linearly from the initial value to the final value over the decay range.
    Where the decay range is the range of steps between the decay start step (inclusive) and decay stop step (exclusive).
    
    Parameters
    ----------
    `initial_value : float` - The initial value of the parameter.
    
    `final_value : float` - The final value of the parameter.
    
    `decay_start : int` - The inclusive start step of the decay range.
    
    `decay_stop : int` - The exclusive stop step of the decay range.
    
    `decay_rate : float` - The decay rate, this is the factor of the decay range to which the parameter approaches its final value.
    For example, a decay constant of 0.5 will result in a line that approaches the final value half way between the decay start and decay end bound.
    
    Returns
    -------
    `(int) -> float` - A linear parameter decay function, returns the parameter value at a given iteration.
    The arguments given to this functions are assigned as attributes to the function instance returned.
    """
    if decay_rate < 0.0:
        raise ValueError(f"Decay constant must be greater than 0.0. Got; {decay_rate}.")
    
    ## Decay is exclusive of the decay end iteration.
    decay_end -= 1
    
    ## Pre-compute variables.
    value_range: float = initial_value - final_value
    decay_range: int = decay_end - decay_start
    decay_range_rated: float = decay_range * decay_rate
    
    def linear_decay(iteration: int) -> float:
        """Linear decay function."""
        decay_iterations: int = min(decay_range, max(0, iteration - decay_start))
        if decay_iterations == 0:
            return initial_value
        elif decay_iterations < decay_range_rated: ## TODO: Support decay and growth.
            return initial_value - (value_range * (decay_iterations / decay_range_rated))
        return final_value
    
    ## Assign arguments as attributes to the function.
    _locals = locals()
    for param in linear_decay_function.__annotations__:
        if param != "return":
            setattr(linear_decay, param, _locals[param])
    
    return linear_decay

def polynomial_decay_function(initial_value: float,
                              final_value: float,
                              decay_start: int,
                              decay_end: int,
                              decay_rate: float = 1.0
                              ) -> Callable[[int], float]:
    """
    Create a polynomial decay function.
    """
    if decay_rate < 0.0:
        raise ValueError(f"Decay constant must be greater than 0.0. Got; {decay_rate}.")
    
    ## Decay is exclusive of the decay end iteration.
    decay_end -= 1
    
    ## Pre-compute variables.
    value_range: float = initial_value - final_value
    decay_range: int = decay_end - decay_start
    decay_range_rated: float = decay_range * decay_rate
    decay_constant = 1.0e-6 ** (1.0 / decay_range_rated)
    
    def polynomial_decay(iteration: int) -> float:
        """Polynomial decay function."""
        decay_iterations: int = min(decay_range, max(0, iteration - decay_start))
        if decay_iterations == 0:
            return initial_value
        elif decay_iterations < decay_range_rated:
            return final_value + (value_range * (decay_constant ** decay_iterations))
        return final_value
    
    ## Assign arguments as attributes to the function.
    _locals = locals()
    for param in linear_decay_function.__annotations__:
        if param != "return":
            setattr(polynomial_decay, param, _locals[param])
    
    return polynomial_decay

# def exponential_decay_function(initial_value: float,
#                                decay_constant: float,
#                                decay_start: int,
#                                decay_end: int,
#                                iterations_limit: int
#                                ) -> Callable[[int], float]:
#     """
#     Create an exponential decay function.
#     """
#     if not (0.0 <= decay_constant <= 1.0):
#         raise ValueError(f"Decay constant must be between 0.0 and 1.0. Got; {decay_constant}.")
    
#     decay_limit: int = min(iterations_limit, decay_end) - 1
    
#     def exponential_decay(iteration: int) -> float:
#         decay_iterations: int = min(decay_limit - decay_start, max(0, iteration - decay_start))
#         return initial_value * math.exp(-((1.0 - decay_constant) * decay_iterations))
    
#     return exponential_decay

def exponential_decay_function(initial_value: float,
                               final_value: float,
                               decay_start: int,
                               decay_end: int,
                               decay_rate: float = 1.0
                               ) -> Callable[[int], float]:
    """
    Create a bounded exponential decay function.
    """
    if decay_rate < 0.0:
        raise ValueError(f"Decay constant must be greater than 0.0. Got; {decay_rate}.")
    
    ## Decay is exclusive of the decay end iteration.
    decay_end -= 1
    
    ## Pre-compute variables.
    value_range: float = initial_value - final_value
    decay_range: int = decay_end - decay_start
    decay_range_rated: float = decay_range * decay_rate
    
    def exponential_decay(iteration: int) -> float:
        """Exponential decay function."""
        decay_iterations: int = min(decay_range, max(0, iteration - decay_start))
        if decay_iterations == 0:
            return initial_value
        elif decay_iterations < decay_range_rated:
            return initial_value - (value_range * (math.log(decay_iterations) / math.log(decay_range_rated)))
        return final_value
    
    return exponential_decay

def sinusoidal_decay_function(initial_value: float,
                              final_value: float,
                              decay_start: int,
                              decay_end: int,
                              decay_rate: float = 1.0
                              ) -> Callable[[int], float]:
    """
    Create a sinusoidal decay function.
    """
    if decay_rate < 0.0:
        raise ValueError(f"Decay constant must be greater than 0.0. Got; {decay_rate}.")
    
    ## Decay is exclusive of the decay end iteration.
    decay_end -= 1
    
    ## Pre-compute variables.
    value_range: float = initial_value - final_value
    decay_range: int = decay_end - decay_start
    decay_range_rated: float = decay_range * decay_rate
    
    def sinusoidal_decay(iteration: int) -> float:
        decay_iterations: int = min(decay_range, max(0, iteration - decay_start))
        if decay_iterations == 0:
            return initial_value
        elif decay_iterations < decay_range_rated:
            return final_value + (value_range * ((math.cos(math.pi * (decay_iterations / decay_range_rated)) / 2.0) + 0.5))
        return final_value
    
    return sinusoidal_decay

def get_decay_function(decay_type: DecayFunctionType,
                       initial_value: float,
                       final_value: float,
                       decay_start: int,
                       decay_end: int,
                       decay_rate: float = 1.0
                       ) -> Callable[[int], float]:
    """
    Create a parameter decay function of the given type.
    
    Parameters
    ----------
    `decay_type : str` - The type of decay function to create:
        - "lin" - Linear decay function.
        - "pol" - Polynomial decay function.
        - "exp" - Exponential decay function.
        - "b-exp" - Bounded exponential decay function.
        - "sin" - Sinusoidal decay function.
    
    `initial_value : float` - The initial value of the parameter.
    
    `decay_constant : float` - The decay constant.
    
    `decay_start : int` - The iteration at which decay starts.
    
    `decay_end : int` - The iteration at which decay ends.
    
    `iterations_limit : int` - The maximum number of iterations.
    
    Returns
    -------
    `(int) -> float` - The decay function.
    Takes the iteration number as argument and returns the parameter value for that iteration.
    
    Raises
    ------
    `ValueError` - If the decay type is not one of the supported types.
    """
    match decay_type:
        case "lin":
            return linear_decay_function(initial_value, final_value, decay_start, decay_end, decay_rate)
        case "pol":
            return polynomial_decay_function(initial_value, final_value, decay_start, decay_end, decay_rate)
        case "exp":
            return exponential_decay_function(initial_value, final_value, decay_start, decay_end, decay_rate)
        case "sin":
            return sinusoidal_decay_function(initial_value, final_value, decay_start, decay_end, decay_rate)
        case _: ValueError(f"Unknown decay type: {decay_type}")
