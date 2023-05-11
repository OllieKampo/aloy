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

__all__ = (
    "DecayFunctionType",
    "linear_decay_function",
    "polynomial_decay_function",
    "exponential_decay_function",
    "sinusoidal_decay_function",
    "get_decay_function"
)

import math
from typing import Callable, Literal, TypeAlias

DecayFunctionType: TypeAlias = Literal["lin", "pol", "exp", "b-exp", "sin"]


def linear_decay_function(
    initial_value: float,
    final_value: float,
    decay_start: int,
    decay_end: int,
    decay_rate: float = 1.0
) -> Callable[[int], float]:
    """
    Create a linear parameter decay function.

    The parameter value changes linearly from the initial value to the final
    value over the decay range. Where the decay range is the range of steps
    between the decay start step (inclusive) and decay stop step (exclusive).

    Parameters
    ----------
    `initial_value : float` - The initial value of the parameter.

    `final_value : float` - The final value of the parameter. If the final
    value is less than the initial value, the parameter value will decrease
    over the decay range. If the final value is greater than the initial
    value, the value will increase.

    `decay_start : int` - The inclusive start step of the decay range.

    `decay_stop : int` - The exclusive stop step of the decay range.

    `decay_rate : float = 1.0` - The decay rate, this is the factor of the
    decay range to which the parameter approaches its final value. For
    example, a decay rate of 0.5 will result in a line that approaches the
    final value half way between the decay start and decay end bound.

    Returns
    -------
    `(int) -> float` - A linear parameter decay function, returns the
    parameter value at a given iteration. The arguments given to this
    functions are assigned as attributes to the function instance returned.

    Raises
    ------
    `ValueError` - If the decay rate is less than 0.0.

    `ValueError` - If the decay start is greater than the decay end.
    """
    if decay_rate < 0.0:
        raise ValueError(
            f"Decay rate must be greater than 0.0. Got; {decay_rate}.")
    if decay_start > decay_end:
        raise ValueError(
            "Decay start must be less than decay end. "
            f"Got; {decay_start} > {decay_end}."
        )

    # Decay is exclusive of the decay end iteration.
    decay_end -= 1

    # Pre-compute variables.
    value_range: float = initial_value - final_value
    decay_range: int = decay_end - decay_start
    decay_range_rated: float = decay_range * decay_rate

    def linear_decay(iteration: int) -> float:
        """Linear decay function."""
        decay_iterations: int = min(
            decay_range, max(0, iteration - decay_start)
        )
        if decay_iterations == 0:
            return initial_value
        elif decay_iterations < decay_range_rated:
            return initial_value - (value_range * (decay_iterations / decay_range_rated))
        return final_value

    # Assign arguments as attributes to the function.
    _locals = locals()
    for param in linear_decay_function.__annotations__:
        if param != "return":
            setattr(linear_decay, param, _locals[param])

    return linear_decay


def polynomial_decay_function(
    initial_value: float,
    final_value: float,
    decay_start: int,
    decay_end: int,
    decay_rate: float = 1.0
) -> Callable[[int], float]:
    """
    Create a polynomial decay function.

    The parameter value changes polynomially from the initial value to the
    final value over the decay range. Where the decay range is the range of
    steps between the decay start step (inclusive) and decay stop step
    (exclusive).

    Parameters
    ----------
    `initial_value: float` - The initial value of the parameter.

    `final_value: float` - The final value of the parameter. If the final
    value is less than the initial value, the parameter value will decrease
    over the decay range. If the final value is greater than the initial
    value, the value will increase.

    `decay_start: int` - The inclusive start step of the decay range.

    `decay_stop: int` - The exclusive stop step of the decay range.

    `decay_rate: float = 1.0` - The decay rate, this is the factor of the
    decay range to which the parameter approaches its final value. For
    example, a decay rate of 0.5 will result in a line that approaches the
    final value half way between the decay start and decay end bound.

    Returns
    -------
    `(int) -> float` - A polynomial parameter decay function, returns the
    parameter value at a given iteration. The arguments given to this
    functions are assigned as attributes to the function instance returned.

    Raises
    ------
    `ValueError` - If the decay rate is less than 0.0.

    `ValueError` - If the decay start is greater than the decay end.
    """
    if decay_rate < 0.0:
        raise ValueError(
            f"Decay rate must be greater than 0.0. Got; {decay_rate}.")
    if decay_start > decay_end:
        raise ValueError(
            "Decay start must be less than decay end. "
            f"Got; {decay_start} > {decay_end}."
        )

    # Decay is exclusive of the decay end iteration.
    decay_end -= 1

    # Pre-compute variables.
    value_range: float = initial_value - final_value
    decay_range: int = decay_end - decay_start
    decay_range_rated: float = decay_range * decay_rate
    decay_constant = 1.0e-6 ** (1.0 / decay_range_rated)

    def polynomial_decay(iteration: int) -> float:
        """Polynomial decay function."""
        decay_iterations: int = min(
            decay_range, max(0, iteration - decay_start)
        )
        if decay_iterations == 0:
            return initial_value
        elif decay_iterations < decay_range_rated:
            return final_value + (value_range * (decay_constant ** decay_iterations))
        return final_value

    # Assign arguments as attributes to the function.
    _locals = locals()
    for param in linear_decay_function.__annotations__:
        if param != "return":
            setattr(polynomial_decay, param, _locals[param])

    return polynomial_decay


def exponential_decay_function(
    initial_value: float,
    final_value: float,
    decay_start: int,
    decay_end: int,
    decay_rate: float = 1.0
) -> Callable[[int], float]:
    """
    Create an exponential decay function.

    The parameter value changes exponentially from the initial value to the
    final value over the decay range. Where the decay range is the range of
    steps between the decay start step (inclusive) and decay stop step
    (exclusive).

    Parameters
    ----------
    `initial_value: float` - The initial value of the parameter.

    `final_value: float` - The final value of the parameter. If the final
    value is less than the initial value, the parameter value will decrease
    over the decay range. If the final value is greater than the initial
    value, the value will increase.

    `decay_start: int` - The inclusive start step of the decay range.

    `decay_stop : int` - The exclusive stop step of the decay range.

    `decay_rate : float = 1.0` - The decay rate, this is the factor of the
    decay range to which the parameter approaches its final value. For
    example, a decay rate of 0.5 will result in a line that approaches the
    final value half way between the decay start and decay end bound.

    Returns
    -------
    `(int) -> float` - A exponential parameter decay function, returns the
    parameter value at a given iteration. The arguments given to this
    functions are assigned as attributes to the function instance returned.

    Raises
    ------
    `ValueError` - If the decay rate is less than 0.0.

    `ValueError` - If the decay start is greater than the decay end.
    """
    if decay_rate < 0.0:
        raise ValueError(
            f"Decay rate must be greater than 0.0. Got; {decay_rate}.")
    if decay_start > decay_end:
        raise ValueError(
            "Decay start must be less than decay end. "
            f"Got; {decay_start} > {decay_end}."
        )

    # Decay is exclusive of the decay end iteration.
    decay_end -= 1

    # Pre-compute variables.
    value_range: float = initial_value - final_value
    decay_range: int = decay_end - decay_start
    decay_range_rated: float = decay_range * decay_rate

    def exponential_decay(iteration: int) -> float:
        """Exponential decay function."""
        decay_iterations: int = min(
            decay_range, max(0, iteration - decay_start)
        )
        if decay_iterations == 0:
            return initial_value
        elif decay_iterations < decay_range_rated:
            return initial_value - (value_range * (math.log(decay_iterations) / math.log(decay_range_rated)))
        return final_value

    # Assign arguments as attributes to the function.
    _locals = locals()
    for param in linear_decay_function.__annotations__:
        if param != "return":
            setattr(exponential_decay, param, _locals[param])

    return exponential_decay


def sinusoidal_decay_function(
    initial_value: float,
    final_value: float,
    decay_start: int,
    decay_end: int,
    decay_rate: float = 1.0
) -> Callable[[int], float]:
    """
    Create a sinusoidal decay function.

    The parameter value changes along a sinusoidal curve from the initial
    value to the final value over the decay range. Where the decay range is
    the range of steps between the decay start step (inclusive) and decay stop
    step (exclusive).

    Parameters
    ----------
    `initial_value: float` - The initial value of the parameter.

    `final_value: float` - The final value of the parameter. If the final
    value is less than the initial value, the parameter value will decrease
    over the decay range. If the final value is greater than the initial
    value, the value will increase.

    `decay_start: int` - The inclusive start step of the decay range.

    `decay_stop: int` - The exclusive stop step of the decay range.

    `decay_rate: float = 1.0` - The decay rate, this is the factor of the
    decay range to which the parameter approaches its final value. For
    example, a decay rate of 0.5 will result in a line that approaches the
    final value half way between the decay start and decay end bound.

    Returns
    -------
    `(int) -> float` - A sinusoidal parameter decay function, returns the
    parameter value at a given iteration. The arguments given to this
    functions are assigned as attributes to the function instance returned.

    Raises
    ------
    `ValueError` - If the decay rate is less than 0.0.

    `ValueError` - If the decay start is greater than the decay end.
    """
    if decay_rate < 0.0:
        raise ValueError(
            f"Decay rate must be greater than 0.0. Got; {decay_rate}.")
    if decay_start > decay_end:
        raise ValueError(
            "Decay start must be less than decay end. "
            f"Got; {decay_start} > {decay_end}."
        )

    # Decay is exclusive of the decay end iteration.
    decay_end -= 1

    # Pre-compute variables.
    value_range: float = initial_value - final_value
    decay_range: int = decay_end - decay_start
    decay_range_rated: float = decay_range * decay_rate

    def sinusoidal_decay(iteration: int) -> float:
        """Sinusoidal decay function."""
        decay_iterations: int = min(
            decay_range, max(0, iteration - decay_start)
        )
        if decay_iterations == 0:
            return initial_value
        elif decay_iterations < decay_range_rated:
            return (final_value + (value_range * ((math.cos(math.pi * (decay_iterations / decay_range_rated)) / 2.0) + 0.5)))
        return final_value

    # Assign arguments as attributes to the function.
    _locals = locals()
    for param in linear_decay_function.__annotations__:
        if param != "return":
            setattr(sinusoidal_decay, param, _locals[param])

    return sinusoidal_decay


def get_decay_function(
    decay_type: DecayFunctionType,
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
    `decay_type : str | None` - The type of decay function to create:
        - "lin" - Linear decay function.
        - "pol" - Polynomial decay function.
        - "exp" - Exponential decay function.
        - "sin" - Sinusoidal decay function.
    If None, then a indentity function is created, which always returns the
    initial value.

    `initial_value: float` - The initial value of the parameter.

    `final_value: float` - The final value of the parameter.
    If the final value is less than the initial value, the parameter value
    will decrease over the decay range. If the final value is greater than the
    initial value, the value will increase.

    `decay_start: int` - The inclusive start step of the decay range.

    `decay_end: int` - The exclusive stop step of the decay range.

    `decay_rate: float = 1.0` - The decay rate, this is the factor of the
    decay range to which the parameter approaches its final value. For
    example, a decay rate of 0.5 will result in a line that approaches
    the final value half way between the decay start and decay end bound.

    Returns
    -------
    `(int) -> float` - The decay function, returns the parameter value at a
    given iteration.

    Raises
    ------
    `ValueError` - If the decay type is not one of the supported types.

    `ValueError` - If the decay rate is less than 0.0.

    `ValueError` - If the decay start is greater than the decay end.
    """
    match decay_type:
        case "lin":
            return linear_decay_function(
                initial_value, final_value, decay_start, decay_end, decay_rate
            )
        case "pol":
            return polynomial_decay_function(
                initial_value, final_value, decay_start, decay_end, decay_rate
            )
        case "exp":
            return exponential_decay_function(
                initial_value, final_value, decay_start, decay_end, decay_rate
            )
        case "sin":
            return sinusoidal_decay_function(
                initial_value, final_value, decay_start, decay_end, decay_rate
            )
        case _: ValueError(f"Unknown decay type: {decay_type}")

    def identity_decay(iteration: int) -> float:
        """Identity decay function."""
        return initial_value

    # Assign arguments as attributes to the function.
    _locals = locals()
    for param in get_decay_function.__annotations__:
        if param not in ["decay_type", "return"]:
            setattr(identity_decay, param, _locals[param])

    return identity_decay
