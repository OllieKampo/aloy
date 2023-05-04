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

from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
import inspect
from numbers import Real
import threading
import types
from typing import (
    Callable, Final, Iterable, Iterator, KeysView, Literal, Mapping,
    NamedTuple, Optional, SupportsFloat, TypeAlias, TypeVar, final
)
import statistics

from control.controlutils import ControllerTimer
from datastructures.graph import Graph
from datastructures.mappings import TwoWayMap
from datastructures.views import SetView

__copyright__ = "Copyright (C) 2022 Oliver Michael Kamperis"
__license__ = "GPL-3.0"

__all__ = (
    "Controller",
    "ControllerCombiner",
    "ControlledSystem",
    "SystemController",
    "AutoSystemController"
)


def __dir__() -> tuple[str, ...]:
    """Get the names of the module attributes."""
    return __all__


ET = TypeVar("ET", bound=SupportsFloat)


def clamp(
    value: ET,
    max_: ET | None,
    min_: ET | None, /
) -> ET:
    """
    Clamp the given value to the specified minimum and maximum values.

    If the maximum or minimum values are `None` then they are ignored.
    """
    if max_ is not None and value > max_:  # type: ignore
        return max_
    if min_ is not None and value < min_:  # type: ignore
        return min_
    return value


def calc_error(
    control_input: ET,
    setpoint: ET, /,
    max_: ET | None = None,
    min_: ET | None = None
) -> ET:
    """
    Get the error between an input and a setpoint.

    If the maximum or minimum values are given and not `None` then clamp
    the control input to the specified range before calculating the error.
    """
    control_input = clamp(control_input, max_, min_)
    return control_input - setpoint  # type: ignore


class Controller(metaclass=ABCMeta):
    """
    Base class for controller classes.

    The abstract methods `control_output()` and `reset()` must be implemented
    by subclasses. It is also the responsibility of subclasses to update the
    `_latest_input`, `_latest_error`, and `_latest_output` protected instance
    variables when calculating the control output, and to call the super-class
    `reset()` method when resetting the controller.
    """

    __slots__ = {
        "__input_limits": "The controller input limits.",
        "__output_limits": "The controller output limits.",
        "__inp_trans": "The control input transform function.",
        "__err_trans": "The control error transform function.",
        "__out_trans": "The control output transform function.",
        "__initial_error": "The initial value of the error, None if unknown.",
        "_latest_input": "The latest control input.",
        "_latest_error": "The latest control error.",
        "_latest_output": "The latest control output."
    }

    def __init__(
        self,
        input_limits: tuple[float | None, float | None] = (None, None),
        output_limits: tuple[float | None, float | None] = (None, None),
        input_trans: Callable[[float], float] | None = None,
        error_trans: Callable[[float], float] | None = None,
        output_trans: Callable[[float], float] | None = None,
        initial_error: float | None = None
    ) -> None:
        """
        Create a new controller.

        Parameters
        ----------
        `input_limits : tuple[float | None, float | None]` - Input limits,
        (lower, upper). See `jinx.control.controllers.clamp()` for details.

        `output_limits : tuple[float | None, float | None]` - Output limits,
        (lower, upper). See `jinx.control.controllers.clamp()` for details.

        `input_trans : Callable[[float], float]` - Input transform function.
        This function is applied to the control input before calculating error.

        `error_trans : Callable[[float], float]` - Error transform function.
        This function is applied to the control error before calculating the
        control output.

        `output_trans : Callable[[float], float]` - Output transform function.
        This function is applied to the control output before returning it.

        `initial_error : float | None` - The initial value of the error, None
        if unknown. If None, the error value calculated on the first call to
        `control_output()` is used as the initial error, and resultantly the
        integral and derivative terms cannot be calculated until the second
        call (since they require at least two data points).
        """
        # Controller input and output limits.
        if not isinstance(input_limits, tuple) or len(input_limits) != 2:
            raise ValueError("Input limits must be a 2-tuple. "
                             f"Got; {input_limits}.")
        if not isinstance(output_limits, tuple) or len(output_limits) != 2:
            raise ValueError("Output limits must be a 2-tuple. "
                             f"Got; {output_limits}.")
        self.__input_limits: tuple[float | None, float | None] = input_limits
        self.__output_limits: tuple[float | None, float | None] = output_limits

        # Controller input, error, and output transform functions.
        ct = self.__check_transform
        self.__inp_trans: Callable[[float], float] | None
        self.__inp_trans = ct("Input", input_trans)
        self.__err_trans: Callable[[float], float] | None
        self.__err_trans = ct("Error", error_trans)
        self.__out_trans: Callable[[float], float] | None
        self.__out_trans = ct("Output", output_trans)

        # Keep record of initial error for resetting.
        self.__initial_error: float | None = initial_error

        # Keep record of latest input, error, and output.
        self._latest_input: float | None = None
        self._latest_error: float | None = initial_error
        self._latest_output: float | None = None

    @staticmethod
    def __check_transform(
        name: str,
        transform: Callable[[ET], ET] | None
    ) -> Callable[[ET], ET] | None:
        """
        Check that the given transform function is valid.

        The transform function must be a callable with one argument.
        """
        if transform is None:
            return None
        if (not callable(transform) or
                len(inspect.signature(transform).parameters) != 1):
            raise ValueError(f"{name} transform must be a function "
                             f"with one argument. Got; {transform}.")
        return transform

    @property
    def input_limits(self) -> tuple[float | None, float | None]:
        """
        Get or set the input limits.

        `input_limits : (float | None, float | None)` - The input limits.
        """
        return self.__input_limits

    @input_limits.setter
    def input_limits(
        self,
        input_limits: tuple[float | None, float | None]
    ) -> None:
        """Set the input limits, (lower, upper)."""
        self.__input_limits = input_limits

    @property
    def output_limits(self) -> tuple[float | None, float | None]:
        """
        Get or set the output limits.

        `output_limits : (float | None, float | None)` - The output limits.
        """
        return self.__output_limits

    @output_limits.setter
    def output_limits(
        self,
        output_limits: tuple[float | None, float | None]
    ) -> None:
        """Set the output limits, (lower, upper)."""
        self.__output_limits = output_limits

    @property
    def input_transform(self) -> Callable[[float], float] | None:
        """Get the input transform function."""
        return self.__inp_trans

    @input_transform.setter
    def input_transform(
        self,
        input_transform: Callable[[float], float] | None
    ) -> None:
        """Set the input transform function."""
        self.__inp_trans = input_transform

    @input_transform.deleter
    def input_transform(self) -> None:
        """Remove the input transform function."""
        self.__inp_trans = None

    @property
    def error_transform(self) -> Callable[[float], float] | None:
        """Get the error transform function."""
        return self.__err_trans

    @error_transform.setter
    def error_transform(
        self,
        error_transform: Callable[[float], float] | None
    ) -> None:
        """Set the error transform function."""
        self.__err_trans = error_transform

    @error_transform.deleter
    def error_transform(self) -> None:
        """Remove the error transform function."""
        self.__err_trans = None

    @property
    def output_transform(self) -> Callable[[float], float] | None:
        """Get the output transform function."""
        return self.__out_trans

    @output_transform.setter
    def output_transform(
        self,
        output_transform: Callable[[float], float] | None
    ) -> None:
        """Set the output transform function."""
        self.__out_trans = output_transform

    @output_transform.deleter
    def output_transform(self) -> None:
        """Remove the output transform function."""
        self.__out_trans = None

    def transform_input(self, input_value: float) -> float:
        """Transform the input value."""
        if self.__inp_trans is None:
            return input_value
        return self.__inp_trans(input_value)

    def transform_error(self, error_value: float) -> float:
        """Transform the error value."""
        if self.__err_trans is None:
            return error_value
        return self.__err_trans(error_value)

    def transform_output(self, output_value: float) -> float:
        """Transform the output value."""
        if self.__out_trans is None:
            return output_value
        return self.__out_trans(output_value)

    @property
    def initial_error(self) -> float | None:
        """
        Get or set the initial error value.

        `initial_error : {float | None}` - The initial value of the error.
        Returns None if unknown.
        """
        return self.__initial_error

    @initial_error.setter
    def initial_error(self, initial_error: float | None) -> None:
        """Set the initial error."""
        self.__initial_error = initial_error

    @property
    def latest_input(self) -> float | None:
        """
        Get the latest control input.

        Returns None if no input has been given since the last reset.
        """
        return self._latest_input

    @property
    def latest_error(self) -> float | None:
        """
        Get the latest control error.

        Returns None if no error has been calculated since the last reset.
        """
        return self._latest_error

    @property
    def latest_output(self) -> float | None:
        """
        Get the latest control output.

        Returns None if no output has been calculated since the last reset.
        """
        return self._latest_output

    @abstractmethod
    def control_output(
        self,
        control_input: float,
        setpoint: float, /,
        delta_time: float,
        abs_tol: float | None = None
    ) -> float:
        """
        Get the control output.

        Parameters
        ----------
        `control_input : float` - The control input
        (the measured value of the control variable).

        `setpoint : float` - The control setpoint
        (the desired value of the control variable).

        `delta_time : float` - The time difference since the last call.

        `abs_tol : float` - The absolute tolerance for the time difference.

        Returns
        -------
        `float` - The control output.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Reset the controller state."""
        self._latest_input = None
        self._latest_error = self.__initial_error
        self._latest_output = 0.0


__COMBINFER_FUNCS: Final[dict[str, Callable[[Iterable[float]], float]]] = {
    "sum": sum,
    "mean": statistics.mean,
    "median": statistics.median
}


def __get_combiner_func(
    mode: Literal["sum", "mean", "median"] = "mean"
) -> Callable[[Iterable[float]], float]:
    """
    Get the combiner function for a given mode.

    Parameters
    ----------
    `mode : {"sum", "mean", "median"}` - The output combination mode.

    Returns
    -------
    `Callable[[Iterable[float]], float]` - The combiner function.
    """
    if mode not in __COMBINFER_FUNCS:
        raise ValueError(f"Invalid mode {mode!r}. "
                         f"Choose from {tuple(__COMBINFER_FUNCS)!r}.")
    return __COMBINFER_FUNCS[mode]


@final
class ControllerCombiner(Controller):
    """
    Class defining controller combiners.

    A controller combiner is a controller which combines the output of
    multiple other "inner" controllers in parallel, to produce one final
    output.
    """

    __slots__ = {
        "__inner_controllers": "The inner controllers.",
        "__mode": "The output combination mode.",
        "__weights": "The output combination weights of inner controllers.",
        "__weights_updater": "The callback to update the weights."
    }

    def __init__(
        self,
        controllers: Mapping[str, Controller],
        mode: Literal["sum", "mean", "median"] = "mean",
        weights: Mapping[str, float] | None = None,
        weight_updater: Callable[[Mapping[str, float]], None] | None = None,
        input_limits: tuple[float | None, float | None] = (None, None),
        output_limits: tuple[float | None, float | None] = (None, None),
        input_trans: Callable[[float], float] | None = None,
        error_trans: Callable[[float], float] | None = None,
        output_trans: Callable[[float], float] | None = None,
        initial_error: float | None = None
    ) -> None:
        """
        Create a controller combiner from a mapping of controllers.

        Parameters
        ----------
        `controllers: Mapping[str, Controller]` - The inner controllers.
        Given as a mapping of controller names to controller instances.

        `mode: "sum" | "mean" | "median" = "mean"` - The output combination
        mode of the inner controller outputs.

        `weights: Mapping[str, float] | None = None` - The output combination
        weights of the inner controllers. Given as a mapping of controller
        names to weights. If None, then all inner controllers get equal weight.

        `weight_updater:
        Callable[[float, float, Mapping[str, float]], Mapping[str, float]]` -
        The callback to update the weights. The signature is:
        `callback(error, latest_output, weights) -> updated_weights`.
        The callback is called every time the control output is calculated,
        with the error to the setpoint, latest output, and the current weights
        as arguments, and must return updated weights. If a weight is not
        present in the returned mapping, then its old value is preserved.
        """
        super().__init__(
            input_limits, output_limits,
            input_trans, error_trans, output_trans,
            initial_error
        )
        if not isinstance(controllers, Mapping):
            raise TypeError("Controllers must be a mapping. "
                            f"Got; {controllers!r} of {type(controllers)!r}.")
        self.__inner_controllers: dict[str, Controller] = dict(controllers)

        self.__mode: Callable[[Iterable[float]], float]
        self.mode = mode

        self.__weights: dict[str, float]
        self.weights = weights  # type: ignore

        self.__weights_updater: Callable[[str, float], None] | None
        self.weights_updater = weight_updater  # type: ignore

    @property
    def inner_controllers(self) -> types.MappingProxyType[str, Controller]:
        """Get the inner controllers."""
        return types.MappingProxyType(self.__inner_controllers)

    @property
    def mode(self) -> str:
        """Get the output combination mode of the inner controllers."""
        return self.__mode.__name__

    @mode.setter
    def mode(self, mode: Literal["sum", "mean", "median"]) -> None:
        """Set the output combination mode of the inner controllers."""
        self.__mode = __get_combiner_func(mode)

    @property
    def weights(self) -> types.MappingProxyType[str, float]:
        """Get the output combination weights of the inner controllers."""
        return types.MappingProxyType(self.__weights)

    @weights.setter
    def weights(self, weights: Mapping[str, float] | None) -> None:
        """Set the output combination weights of the inner controllers."""
        if weights is None:
            self.__weights = {name: 1.0 for name in self.__inner_controllers}
        else:
            if not isinstance(weights, Mapping):
                raise TypeError("Weights must be a mapping or None. "
                                f"Got; {weights!r} of {type(weights)!r}.")
            for name in weights:
                if name not in self.__inner_controllers:
                    raise ValueError(f"Invalid weight name {name!r}.")
            self.__weights |= weights

    def get_controller(self, name: str) -> Controller:
        """
        Get an inner controller.

        Parameters
        ----------
        `name : str` - The name of the controller.

        Returns
        -------
        `Controller` - The controller instance.
        """
        return self.__inner_controllers[name]

    def add_controller(
        self,
        name: str,
        controller: Controller,
        weight: float = 1.0
    ) -> None:
        """
        Add an inner controller.

        Parameters
        ----------
        `name: str` - The name of the controller.

        `controller: Controller` - The controller instance.

        `weight: float = 1.0` - The output combination weight of the
        controller.

        Notes
        -----
        If the controller name already exists, then it is replaced.
        """
        self.__inner_controllers[name] = controller
        self.__weights[name] = weight

    def remove_controller(self, name: str) -> None:
        """
        Remove an inner controller.

        Parameters
        ----------
        `name : str` - The name of the controller.
        """
        del self.__inner_controllers[name]
        del self.__weights[name]

    def update_weights(self, weights: Mapping[str, float]) -> None:
        """
        Update the output combination weights of the inner controllers.

        Parameters
        ----------
        `weights : Mapping[str, float]` - The output combination weights of the
        inner controllers, given as a mapping of controller names to weights.
        If a weight is not present in the mapping, then its old value is kept.
        """
        if not isinstance(weights, Mapping):
            raise TypeError("Weights must be a mapping. "
                            f"Got; {weights!r} of {type(weights)!r}.")
        self.__weights.update(weights)

    @property
    def weights_updater(
        self
    ) -> Callable[[float, float, Mapping[str, float]], Mapping[str, float]] | None:
        """Get the callback to update the weights."""
        return self.__weights_updater

    @weights_updater.setter
    def weights_updater(
        self,
        callback: Callable[[float, float, Mapping[str, float]], Mapping[str, float]] | None
    ) -> None:
        """Set a callback to update the output combination weights of the inner controllers."""
        self.__weights_updater = callback

    def control_output(
        self,
        control_input: float,
        setpoint: float, /,
        delta_time: float,
        abs_tol: float | None = None
    ) -> float:
        """
        Get the combined control output of the inner controllers.

        Parameters
        ----------
        `control_input : float` - The control input
        (the measured value of the control variable).

        `setpoint : float` - The control setpoint
        (the desired value of the control variable).

        `delta_time : float` - The time difference since the last call.

        `abs_tol : float` - The absolute tolerance for the time difference.

        Returns
        -------
        `float` - The combined control output of the inner controllers.
        """
        self._latest_input = control_input
        error: float = calc_error(control_input, setpoint)
        self._latest_error = error
        if self.__weights_updater is not None:
            weights = self.__weights_updater(
                error, self._latest_output, self.weights
            )
            self.update_weights(weights)
        outputs: list[float] = [
            (controller.control_output(
                control_input, setpoint,
                delta_time, abs_tol
            ) * self.__weights[name])
            for name, controller
            in self.__inner_controllers.items()
        ]
        self._latest_output = self.__mode(outputs)
        return self._latest_output

    def reset(self) -> None:
        """Reset all inner controllers."""
        for controller in self.__inner_controllers.values():
            controller.reset()


class MultiVariateController:
    """
    Class defining a multi-variate controller.

    Multi-variate controllers are composed of many controllers and can
    handle multiple inputs and outputs with complex mappings.
    """

    __slots__ = {
        "__modules": "The modules of the controller, one for each output.",
        "__input_output_mapping": "The mapping of inputs to outputs.",
        "__weights": "The weights of the inputs for each output.",
        "__modes": "The modes of the outputs."
    }

    def __init__(self) -> None:
        """Create a moduler controller from a mapping of controllers."""
        # Maps: output_name -> input_name -> controller
        self.__modules: dict[str, dict[str, Controller]] = {}

        # Maps: input_name -> output_name
        self.__input_output_mapping = TwoWayMap[str]()

        # Maps: output_name -> input_name -> weight of controller to output
        self.__weights: dict[str, dict[str, float]] = {}

        # Maps: output_name -> mode of output
        self.__modes: dict[str, Callable[[Iterable[float]], float]] = {}

    @property
    def inputs(self) -> KeysView[str]:
        """Get the input names."""
        return self.__input_output_mapping.forwards.keys()

    @property
    def outputs(self) -> KeysView[str]:
        """Get the output names."""
        return self.__input_output_mapping.backwards.keys()

    def control_inputs_for(self, output_name: str) -> SetView[str]:
        """Get the input names for a given output module name."""
        return self.__input_output_mapping.backwards[output_name]

    def control_outputs_for(self, input_name: str) -> SetView[str]:
        """Get the control output names for the moduler controller."""
        return self.__input_output_mapping.forwards[input_name]

    def declare_output(
        self,
        output_name: str,
        mode: Literal["sum", "mean", "median"] = "mean"
    ) -> None:
        """
        Declare a module.

        Parameters
        ----------
        `output_name: str` - The name of the output.

        `mode: Literal["sum", "mean", "median"] = "mean"` - The mode of the
        output. If "sum", then the sum of the outputs is returned. If "mean",
        then the mean of the outputs is returned. If "median", then the median
        of the outputs is returned.
        """
        if output_name in self.__modules:
            raise ValueError(
                f"Module for Output {output_name!r} already exists."
            )
        self.__modules[output_name] = {}
        self.__weights[output_name] = {}
        self.__modes[output_name] = __get_combiner_func(mode)

    def add_module(
        self,
        output_name: str,
        controllers: Mapping[str, Controller],
        weights: Mapping[str, float] | None = None,
        mode: Literal["sum", "mean", "median"] = "mean"
    ) -> None:
        """
        Add a module to the moduler controller.
        """
        if output_name in self.__modules:
            raise ValueError(
                f"Module for output {output_name!r} already exists. "
                "Use `update_module` instead."
            )

        for input_name, controller in controllers.items():
            self.__input_output_mapping.add(input_name, output_name)

            if not isinstance(controller, Controller):
                raise TypeError(
                    f"Controllers must be of type {Controller!r}. "
                    f"Got; {controller!r} of {type(controller)!r}."
                )
            self.__modules[output_name][input_name] = controller

        if weights is None:
            self.__weights[output_name] = {name: 1.0 for name in controllers}
        else:
            self.__weights[output_name] = dict(weights)
        self.__modes[output_name] = __get_combiner_func(mode)

    def update_module(
        self,
        output_name: str,
        controllers: Mapping[str, Controller],
        weights: Mapping[str, float] | None = None,
        mode: Literal["sum", "mean", "median"] | None = None
    ) -> None:
        """
        Update a module in the moduler controller.
        """
        if output_name not in self.__modules:
            raise ValueError(
                f"Module for output {output_name!r} does not exist. "
                "Use `add_module` instead."
            )

        for input_name, controller in controllers.items():
            if output_name not in self.__input_output_mapping[input_name]:
                self.__input_output_mapping.add(input_name, output_name)

            if not isinstance(controller, Controller):
                raise TypeError(
                    f"Controllers must be of type {Controller!r}. "
                    f"Got; {controller!r} of {type(controller)!r}."
                )
            self.__modules[output_name][input_name] = controller

        if weights is None:
            for name in controllers:
                if name not in self.__weights[output_name]:
                    self.__weights[output_name][name] = 1.0
        else:
            self.__weights[output_name] = dict(weights)
        if mode is not None:
            self.__modes[output_name] = __get_combiner_func(mode)

    def remove_module(self, output_name: str, /) -> None:
        """Remove a module and all of its controllers."""
        if output_name not in self.__modules:
            raise ValueError(
                f"Module for output {output_name!r} does not exist."
            )
        del self.__modules[output_name]
        del self.__input_output_mapping[output_name]
        del self.__weights[output_name]
        del self.__modes[output_name]

    def get_controller(
        self,
        input_name: str,
        output_name: str, /
    ) -> Controller:
        """Get a controller for a given input and output mapping."""
        return self.__modules[output_name][input_name]

    def add_controller(
        self,
        input_name: str,
        output_name: str,
        controller: Controller,
        weight: float = 1.0
    ) -> None:
        """
        Add a controller to the moduler controller.
        """
        if output_name not in self.__modules:
            raise ValueError(
                f"Module for output {output_name!r} does not exist."
                "Use `declare_module` to declare a module."
            )
        if input_name in self.__input_output_mapping.forwards:
            raise ValueError(
                f"Input {input_name!r} already exists as an output."
            )
        if self.__input_output_mapping.maps_to(input_name, output_name):
            raise ValueError(
                f"Input {input_name!r} already maps to output {output_name!r}."
            )
        if not isinstance(controller, Controller):
            raise TypeError(
                f"Controller must be of type {Controller!r}. "
                f"Got; {controller!r} of {type(controller)!r}."
            )
        self.__modules[output_name][input_name] = controller
        self.__input_output_mapping.add(input_name, output_name)
        self.__weights[output_name][input_name] = weight

    def remove_controller(self, input_name: str, output_name: str, /) -> None:
        """
        Remove a controller from the moduler controller.

        If the module for the given output is empty after the removal, the
        module is also removed.
        """
        del self.__modules[output_name][input_name]
        del self.__input_output_mapping[(input_name, output_name)]
        del self.__weights[output_name][input_name]
        if not self.__modules[output_name]:
            self.remove_module(output_name)

    def update_weights(
        self,
        output_name: str,
        weights: Mapping[str, float]
    ) -> None:
        """
        Update the output combination weights of the inner controllers.
        """
        self.__weights[output_name].update(weights)

    def control_output(
        self,
        control_inputs: Mapping[str, float],
        setpoints: Mapping[str, float], /,
        delta_time: float,
        abs_tol: float | None = None
    ) -> dict[str, float]:
        """
        Get the combined control output of the inner controllers.

        All control input values must be provided, but only non-cascaded input
        set-points should be provided. The set-points for cascaded inputs are
        the outputs of the controllers they are cascaded from.
        """
        outputs: dict[str, float] = dict.fromkeys(self.__modules.keys(), 0.0)
        for output_name, module in self.__modules.items():
            output: list[float] = []
            for input_name, controller in module.items():
                if input_name not in control_inputs:
                    raise KeyError(f"Control input variable {input_name!r}"
                                   f"not found. Got; {control_inputs!r}.")
                output.append(
                    controller.control_output(
                        control_inputs[input_name],
                        setpoints[input_name],
                        delta_time,
                        abs_tol
                    ) * self.__weights[output_name][input_name]
                )
            outputs[output_name] = self.__modes[output_name](output)
        return outputs


class CascadingController:
    """
    Class defining moduler controllers.

    Moduler controllers are composed of many controllers and can
    handle multiple inputs and outputs with complex mappings.
    Controllers are bundled into modules, which are then combined
    in series or in parallel to form the final controller.

    Moduler controllers can also be used to define cascading controllers,
    in which the output of one controller is the setpoint of another
    controller.
    """

    __slots__ = {
        "__modules": "The modules of the controller.",
        "__input_output_mapping": "The mapping of inputs to outputs.",
        "__output_input_cascade_mapping": "The mapping of outputs to inputs "
                                          "for cascaded controllers.",
        "__weights": "The weights of the modules.",
        "__modes": "The modes of the outputs."
    }

    def __init__(self) -> None:
        """Create a moduler controller from a mapping of controllers."""
        # Maps: output_name -> input_name -> controller
        self.__modules: dict[str, dict[str, Controller]] = {}

        # Maps: input_name -> output_name
        # TODO: Change to Graph or LayerMap? We need to be able to calculate
        # control outputs in the correct order, starting from layer 1 inputs.
        self.__input_output_mapping = Graph[str]()

        # Maps: output_name -> input_name -> weight of controller to output
        self.__weights: dict[str, dict[str, float]] = {}

        # Maps: output_name -> mode of output
        self.__modes: dict[str, Callable[[Iterable[float]], float]] = {}

    @property
    def all_inputs(self) -> KeysView[str]:
        """Get the input names."""
        return self.__input_output_mapping.forwards.keys()

    @property
    def all_outputs(self) -> KeysView[str]:
        """Get the output names."""
        return self.__input_output_mapping.backwards.keys()

    @property
    def cascaded_inputs(self) -> KeysView[str]:
        """Get the cascaded input names."""
        return self.__output_input_cascade_mapping.backwards.keys()

    @property
    def cascaded_outputs(self) -> KeysView[str]:
        """Get the cascaded output names."""
        return self.__output_input_cascade_mapping.forwards.keys()

    @property
    def control_inputs(self) -> KeysView[str]:
        """Get the control input names."""
        return self.__input_output_mapping.forwards.keys()

    @property
    def control_outputs(self) -> KeysView[str]:
        """Get the control output names."""
        return self.all_outputs - self.cascaded_outputs

    @property
    def setpoint_inputs(self) -> KeysView[str]:
        """Get the setpoint input names."""
        return self.all_inputs

    def control_inputs_for(self, output_name: str) -> list[str]:
        """Get the input names for a given output module name."""
        return self.__input_output_mapping.backwards[output_name]

    def control_outputs_for(self, input_name: str) -> list[str]:
        """Get the control output names for the moduler controller."""
        return self.__input_output_mapping.forwards[input_name]

    def declare_module(
        self,
        output_name: str,
        inputs: Iterable[str],
        mode: Literal["sum", "mean", "median"] = "mean"
    ) -> None:
        """
        Declare a module.

        Parameters
        ----------
        `output_name: str` - The name of the output.

        `inputs: Iterable[str]` - The names of the inputs.

        `mode: Literal["sum", "mean", "median"] = "mean"` - The mode of the
        output. If "sum", then the sum of the outputs is returned. If "mean",
        then the mean of the outputs is returned. If "median", then the median
        of the outputs is returned.
        """
        if output_name in self.__modules:
            raise ValueError(f"Module for Output {output_name!r} already "
                             "exists.")

        if not isinstance(inputs, Iterable):
            raise TypeError(f"Inputs must be iterable. "
                            f"Got; {inputs!r} of {type(inputs)!r}.")

        self.__modules[output_name] = {}
        for input_name in inputs:
            self.__input_output_mapping.add_edge(input_name, output_name)
            self.__input_output_mapping.tag_vertex(input_name, 0)
            self.__input_output_mapping.tag_vertex(output_name, -1)
        self.__weights[output_name] = {}
        self.__modes[output_name] = __get_combiner_func(mode)

    def add_module(
        self,
        output_name: str,
        controllers: Mapping[str, Controller],
        weights: Mapping[str, float] | None = None,
        mode: Literal["sum", "mean", "median"] = "mean"
    ) -> None:
        """
        Add a module to the moduler controller.
        """
        if output_name in self.__modules:
            raise ValueError(f"Module for output {output_name!r} already "
                             "exists. Use `update_module` instead.")

        for input_name, controller in controllers.items():
            self.__input_output_mapping.add_edge(input_name, output_name)

            if not isinstance(controller, Controller):
                raise TypeError(f"Controllers must be of type {Controller!r}. "
                                f"Got; {controller!r} of {type(controller)!r}.")
            self.__modules[output_name][input_name] = controller

        if weights is None:
            self.__weights[output_name] = {name: 1.0 for name in controllers}
        else:
            self.__weights[output_name] = dict(weights)
        self.__modes[output_name] = __get_combiner_func(mode)

    def update_module(
        self,
        output_name: str,
        controllers: Mapping[str, Controller],
        weights: Mapping[str, float] | None = None,
        mode: Literal["sum", "mean", "median"] | None = None
    ) -> None:
        """
        Update a module in the moduler controller.
        """
        if output_name not in self.__modules:
            raise ValueError(f"Module for output {output_name!r} does not "
                             "exist. Use `add_module` instead.")

        for input_name, controller in controllers.items():
            if output_name not in self.__input_output_mapping[input_name]:
                self.__input_output_mapping.add_edge(input_name, output_name)

            if not isinstance(controller, Controller):
                raise TypeError(f"Controllers must be of type {Controller!r}. "
                                f"Got; {controller!r} of {type(controller)!r}.")
            self.__modules[output_name][input_name] = controller

        if weights is None:
            for name in controllers:
                if name not in self.__weights[output_name]:
                    self.__weights[output_name][name] = 1.0
        else:
            self.__weights[output_name] = dict(weights)
        if mode is not None:
            self.__modes[output_name] = __get_combiner_func(mode)

    def remove_module(self, output_name: str, /) -> None:
        """Remove a module and all of its controllers."""
        if output_name not in self.__modules:
            raise ValueError(f"Module for output {output_name!r} does not "
                             "exist.")
        del self.__modules[output_name]
        del self.__input_output_mapping[output_name]
        del self.__weights[output_name]
        del self.__modes[output_name]

    def get_controller(
        self,
        input_name: str,
        output_name: str, /
    ) -> Controller:
        """Get a controller for a given input and output mapping."""
        return self.__modules[output_name][input_name]

    def add_controller(
        self,
        input_name: str,
        output_name: str,
        controller: Controller,
        weight: float = 1.0
    ) -> None:
        """
        Add a controller to the moduler controller.
        """
        if output_name not in self.__modules:
            raise ValueError(f"Module for output {output_name!r} does not exist."
                             "Use `declare_module` to declare a module.")
        if input_name in self.__input_output_mapping.forwards:
            raise ValueError(f"Input {input_name!r} already exists as an output.")
        if self.__input_output_mapping.maps_to(input_name, output_name):
            raise ValueError(f"Input {input_name!r} already maps to "
                             f"output {output_name!r}.")
        if not isinstance(controller, Controller):
            raise TypeError(f"Controller must be of type {Controller!r}. "
                            f"Got; {controller!r} of {type(controller)!r}.")
        self.__modules[output_name][input_name] = controller
        self.__input_output_mapping.add(input_name, output_name)
        self.__weights[output_name][input_name] = weight

    def remove_controller(self, input_name: str, output_name: str, /) -> None:
        """
        Remove a controller from the moduler controller.

        If the module for the given output is empty after the removal, the
        module is also removed.
        """
        del self.__modules[output_name][input_name]
        del self.__input_output_mapping[(input_name, output_name)]
        del self.__weights[output_name][input_name]
        if not self.__modules[output_name]:
            self.remove_module(output_name)

    def update_weights(
        self,
        output_name: str,
        weights: Mapping[str, float]
    ) -> None:
        """
        Update the output combination weights of the inner controllers.
        """
        self.__weights[output_name].update(weights)

    def cascade_output_to_input(
        self,
        output_name: str,
        input_name: str, /
    ) -> None:
        """
        Cascade the output of one controller to the input set-point of another.

        Note that this does not cascade the output of one controller to the
        input of another. It only cascades the output of one controller to the
        set-point of the input of another controller.
        """
        if "cascaded_to" in self.__input_output_mapping.get_vertex_tags(input_name):
            pass  # TODO
        self.__input_output_mapping.untag_vertex(output_name)
        self.__input_output_mapping.untag_vertex(input_name)
        self.__input_output_mapping.tag_vertex(output_name, "cascaded_to", input_name)  # Will not be given as control output, instead cascaded to input set-point
        self.__input_output_mapping.tag_vertex(input_name, "cascaded_from", output_name, 1)  # No input-setpoint need to be given, instead cascaded from output
        self.__input_output_mapping.add_edge(output_name, input_name)

    def control_output(
        self,
        control_inputs: Mapping[str, float],
        setpoints: Mapping[str, float], /,
        delta_time: float,
        abs_tol: float | None = None
    ) -> dict[str, float]:
        """
        Get the combined control output of the inner controllers.

        All control input values must be provided, but only non-cascaded input
        set-points should be provided. The set-points for cascaded inputs are
        the outputs of the controllers they are cascaded from.
        """
        outputs: dict[str, float] = dict.fromkeys(self.__modules.keys(), 0.0)
        for output_name, module in self.__modules.items():
            output: list[float] = []
            for input_name, controller in module.items():
                if input_name not in control_inputs:
                    raise KeyError(f"Control input variable {input_name!r}"
                                   f"not found. Got; {control_inputs!r}.")
                output.append(
                    controller.control_output(
                        control_inputs[input_name],
                        setpoints[input_name],
                        delta_time,
                        abs_tol
                    ) * self.__weights[output_name][input_name]
                )
            outputs[output_name] = self.__modes[output_name](output)
        return outputs


class ControlledSystem:
    """
    Base class mixin for creating controlled system classes.

    A controlled system exposes an interface making it controllable by a
    system controller. The system must expose the following control input
    getting, setpoint getting, and output setting callback methods:
    ```
        get_control_input(var_name: str | None = None) -> float
        get_setpoint(var_name: str | None = None) -> float
        set_control_output(
            output: float,
            delta_time: float,
            var_name: str | None = None
        ) -> None
    ```
    If the system has multiple control variables, the `var_name` argument is
    used to specify which control input/output variable to get/set. In this
    cases, the system can optionally expose the variable names:
    ```
        input_variables: tuple[str] | None
        output_variables: tuple[str] | None
    ```

    This class defines no instance variables and an empty `__slots__`.
    """

    __slots__ = ()

    @property
    def input_variables(self) -> tuple[str] | None:
        """Get the names of the control input variables."""
        return None

    @property
    def output_variables(self) -> tuple[str] | None:
        """Get the names of the control output variables."""
        return None

    @abstractmethod
    def get_control_input(
        self,
        var_name: str | None = None
    ) -> float:
        """Get the control input of the controlled system."""
        raise NotImplementedError

    @abstractmethod
    def get_setpoint(
        self,
        var_name: str | None = None
    ) -> float:
        """Get the setpoint of the controlled system."""
        raise NotImplementedError

    @abstractmethod
    def set_control_output(
        self,
        output: float,
        delta_time: float,
        var_name: str | None = None
    ) -> None:
        """Set the control output to the controlled system."""
        raise NotImplementedError

    def get_input_limits(
        self,
        var_name: str | None = None
    ) -> tuple[float | None, float | None]:
        """Get the control input limits of the controlled system."""
        return (None, None)

    def get_output_limits(
        self,
        var_name: str | None = None
    ) -> tuple[float | None, float | None]:
        """Get the control output limits of the controlled system."""
        return (None, None)


class ControlTick(NamedTuple):
    """
    Named tuple defining a control tick from a system controller.

    Items
    -----
    `ticks: int` - The number of ticks since the last reset.

    `error: float` - The control error of the control tick.

    `output: float` - The control output of the control tick.

    `delta_time: float` - The time difference (in seconds)
    between the last and previous ticks.
    """

    ticks: int
    error: float
    output: float
    delta_time: float


class SystemController:
    """
    Class defining system controllers.

    The `SystemController` does not inherit from the `Controller` class.
    Instead, a system controller encapsulates a standard controller,
    allowing the prior to control that latter.
    """

    __slots__ = {
        "__controller": "The underlying controller.",
        "__system": "The controlled system.",
        "__input_var_name": "The name of the input variable "
                            "to get from the controlled system.",
        "__output_var_name": "The name of the output variable "
                             "to set to the controlled system.",
        "__timer": "The controller timer used to calculate "
                   "time differences between ticks.",
        "__ticks": "The number of ticks since the last reset."
    }

    def __init__(self,
                 controller: Controller,
                 system: ControlledSystem,
                 input_var_name: str | None = None,
                 output_var_name: str | None = None,
                 get_input_limits: bool = False,
                 get_output_limits: bool = False
                 ) -> None:
        """
        Create a system controller from a controller and a controlled system.

        In contrast to the standard controller, a system controller also takes
        a controlled system as input. The control input is taken from, and
        control output set to, the control system, every time the controller
        is 'ticked', handling time dependent calculations automatically.
        Time differences are calculated by an internal timer.

        Parameters
        ----------
        `controller : Controller` - The controller to use.

        `system : ControlledSystem` - The system to control.
        Must implement the `get_control_input(...)`, `get_setpoint(...)`
        and `set_control_output(...)` methods of the `ControlledSystem` mixin.

        `input_var_name: {str | None} = None` - The name of the input
        variable to get from the controlled system.

        `output_var_name: {str | None} = None` - The name of the output
        variable to set to the controlled system.

        `get_input_limits: bool = False` - Whether to get the input limits
        from the controlled system and set them to the controller.

        `get_output_limits: bool = False` - Whether to get the output limits
        from the controlled system and set them to the controller.
        """
        self.__controller: Controller = controller
        self.__system: ControlledSystem = system
        self.__input_var_name: str | None = input_var_name
        self.__output_var_name: str | None = output_var_name
        self.__timer: ControllerTimer = ControllerTimer()
        self.__ticks: int = 0

        if get_input_limits:
            self.__controller.input_limits = \
                self.__system.get_input_limits(input_var_name)
        if get_output_limits:
            self.__controller.output_limits = \
                self.__system.get_output_limits(output_var_name)

    @classmethod
    def from_getsetter(
        cls,
        controller: Controller,
        getter: Callable[[str], float],
        setter: Callable[[float, float, str], None],
        setpoint: Callable[[str], float] | float,
        input_var_name: str | None = None,
        output_var_name: str | None = None
    ) -> "SystemController":
        """
        Create a system controller from getter and setter callbacks.

        This is a convenience method for creating a system controller
        from `get_control_input(...)` and `set_control_output(...)` functions
        that are not attached to a controlled system.

        Parameters
        ----------
        `controller: Controller` - The controller to use.

        `getter: Callable[[str], float]` - The error getter function. Takes
        the name of the error variable as an argument and returns the error.

        `setter: Callable[[float, float, str], None]` - The output setter
        function. Must take the control output, the time difference since
        the last call, and the name of the error variable as arguments.

        `setpoint: Callable[[str], float] | float` - The setpoint function.
        Takes the name of the error variable as an argument and returns the
        setpoint. Alternatively, a constant setpoint can be passed as a float.

        For other parameters, see `jinx.control.controllers.SystemController`.
        """
        if (not callable(getter)
                or len(inspect.signature(getter).parameters) != 1):
            raise TypeError("The getter must be a callable that takes a "
                            "single string argument.")
        if (not callable(setter)
                or len(inspect.signature(setter).parameters) != 3):
            raise TypeError("The setter must be a callable that takes "
                            "three arguments: the control output, the time "
                            "difference since the last call, and the name "
                            "of the error variable.")
        if isinstance(setpoint, float):
            def _setpoint(var_name: str) -> float:
                return setpoint  # type: ignore
            setpoint = _setpoint
        elif (not callable(setpoint)
                or len(inspect.signature(setpoint).parameters) != 1):
            raise TypeError("The setpoint must be a float or a callable "
                            "that takes a single string argument.")
        system = type("getter_setter_system",
                      (ControlledSystem,),
                      {"get_control_input": getter,
                       "set_control_output": setter,
                       "get_setpoint": setpoint})()
        return cls(controller, system, input_var_name, output_var_name)

    def __repr__(self) -> str:
        """Get the string representation of the system controller instance."""
        return f"{self.__class__.__name__}({self.__controller!r}, " \
               f"{self.__system!r}, {self.__input_var_name!r}" \
               f"{self.__output_var_name!r})"

    @property
    def controller(self) -> Controller:
        """Get the controller."""
        return self.__controller

    @property
    def system(self) -> ControlledSystem:
        """Get the controlled system."""
        return self.__system

    @property
    def input_var_name(self) -> str | None:
        """Get the name of the input variable taken from controlled system."""
        return self.__input_var_name

    @property
    def output_var_name(self) -> str | None:
        """Get the name of the output variable set to the controlled system."""
        return self.__output_var_name

    @property
    def ticks(self) -> int:
        """
        Get the number of times this controller has been ticked since the last
        reset.
        """
        return self.__ticks

    def time_since_last_ticked(self, time_factor: float = 1.0) -> float:
        """Get the time in seconds since the controller was last ticked."""
        return self.__timer.time_since_last(time_factor)

    def tick(
        self,
        time_factor: float = 1.0,
        abs_tol: float | None = None
    ) -> ControlTick:
        """
        Tick the controller.

        This calculates the current control output and sets it to the system
        through the system's `set_output(output, delta_time)` callback method.

        Parameters
        ----------
        `time_factor: float = 1.0` - The time factor to use when calculating
        the time difference. The actual time difference is multiplied by this
        value, values smaller than 1.0 will therefore act as if less time has
        passed since the last tick, and vice versa for values greater than 1.0.

        `abs_tol: float | None = None` - The absolute tolerance for the time
        difference, None for no tolerance. The affect of the tolerance is
        dependent on the controller implementation.

        Returns
        -------
        `ControlTick(tick: int, output: float, delta_time: float)` - The tick
        number, the control output, and the time in seconds since the last
        tick (multiplied by the time factor).
        """
        self.__ticks += 1

        controller = self.__controller
        system = self.__system
        input_var_name = self.__input_var_name
        output_var_name = self.__output_var_name

        control_input: float = system.get_control_input(input_var_name)
        setpoint: float = system.get_setpoint(input_var_name)
        delta_time: float = self.__timer.get_delta_time(time_factor)

        output: float = clamp(
            controller.control_output(
                control_input, setpoint, delta_time, abs_tol
            ),
            *system.get_output_limits(output_var_name)
        )
        system.set_control_output(output, delta_time, output_var_name)

        return ControlTick(
            self.__ticks,
            controller.latest_error,  # type: ignore
            output,
            delta_time
        )

    def reset(self) -> None:
        """Reset the controller state."""
        self.__ticks = 0
        self.__timer.reset()
        self.__controller.reset()


# Signature: (iterations, error, output, delta_time, total_time) -> bool
Condition: TypeAlias = Callable[[int, float | None, float, float, float], bool]
# Signature: (iterations, error, output, delta_time, total_time) -> None
DataCallback: TypeAlias = Callable[[int, float, float, float, float], None]


@final
class AutoSystemController:
    """
    Class defining an automatic system controller that runs concurrently in a
    separate thread.

    The contoller can either be ran indefinitely with `run_forever()` until
    an explicit call to `stop()` is made, it can also be ran with loop-like
    stop conditions with `run_for(iterations, time)` or `run_while(condition)`.
    It can similarly be ran in a context manager with
    `context_run_for(iterations, time)` or `context_run_while(condition)`.
    """

    __slots__ = {
        "__system_controller": "The underlying system controller.",
        "__atomic_update_lock": "Lock for atomic updates in run methods.",
        "__in_context": "Whether the controller is currently running in a "
                        "context.",
        "__sleep_time": "The sleep time between control ticks.",
        "__time_factor": "The time factor to use when calculating the time "
                         "difference.",
        "__data_callback": "The data callback function.",
        "__run_forever": "Whether the controller is currently running "
                         "indefinitely.",
        "__condition": "The condition to check for loop-like stop conditions.",
        "__thread": "The thread the controller is running in.",
        "__lock": "Lock for atomic updates in the thread.",
        "__running": "Whether the controller is currently running.",
        "__stopped": "Whether the controller has been stopped."
    }

    def __init__(
        self,
        system_controller: SystemController
    ) -> None:
        """
        Create an automatic system controller.

        An automatic system controller encapsulates a normal system controller,
        allowing it to be ran concurrently in a separate thread of control.

        Parameters
        ----------
        `system_controller: SystemController` - The system controller to
        encapsulate.
        """
        self.__system_controller: SystemController = system_controller

        # Parameters for run methods.
        self.__atomic_update_lock = threading.Lock()
        self.__in_context: bool = False
        self.__sleep_time: float = 0.1
        self.__time_factor: float = 1.0
        self.__run_forever: bool = False
        self.__condition: Condition | None = None

        # Variables for the controller thread.
        self.__thread = threading.Thread(target=self.__run)
        self.__thread.daemon = True
        self.__lock = threading.Lock()
        self.__running = threading.Event()
        self.__stopped = threading.Event()
        self.__data_callback: DataCallback | None = None
        self.__thread.start()

    def __repr__(self) -> str:
        """
        Get an instantiable string representation of the automatic system
        controller instance.
        """
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

            # The lock is held whilst the controller is running.
            with self.__lock:
                stop = self.__stopped.is_set()

                cond = self.__condition
                data = self.__data_callback

                system_controller = self.__system_controller
                controller = system_controller.controller

                error = controller.latest_error
                delta_time, output = 0.0, 0.0
                iterations, total_time = 0, 0.0

                while (not stop
                       and (self.__run_forever
                            or cond(  # type: ignore
                                iterations,
                                error,
                                output,
                                delta_time,
                                total_time
                            )
                            )
                       ):

                    iterations, error, output, delta_time = \
                        system_controller.tick(self.__time_factor)
                    total_time += delta_time
                    if data is not None:
                        data(
                            iterations,
                            error, output,
                            delta_time, total_time
                        )

                    # Preempt stop calls.
                    stop = self.__stopped.wait(self.__sleep_time)

                    # TODO: Add something to handle time errors and catch up.
                    # Keep track of actual tick rate and variance in tick
                    # rate, and number of skipped ticks. If the tick rate is
                    # too high, emit warnings and slow down the tick rate.

            if not stop:
                self.__stop()

    def __check_running_state(self) -> None:
        """
        Check the running state of the controller.

        Raise an error if the controller is running.
        """
        if self.__in_context:
            raise RuntimeError("Cannot run an AutoSystemController while it "
                               "is running in a context manager.")
        if self.is_running:
            raise RuntimeError("Cannot run an AutoSystemController while it "
                               "is already running.")

    def __set_parameters(
        self,
        tick_rate: int,
        time_factor: float,
        data_callback: DataCallback | None,
        run_forever: bool,
        condition: Condition | None
    ) -> None:
        """Set the parameters of the controller."""
        if tick_rate <= 0:
            raise ValueError("Tick rate must be greater than 0. "
                             f"Got; {tick_rate}.")
        if time_factor <= 0.0:
            raise ValueError("Time factor must be greater than 0.0. "
                             f"Got; {time_factor}.")

        self.__sleep_time = (1.0 / tick_rate) / time_factor
        self.__time_factor = time_factor

        # Check and set the data callback.
        if data_callback is not None:
            if not callable(data_callback):
                raise TypeError("The data callback must be a callable.")
            num_params = len(inspect.signature(data_callback).parameters)
            if num_params != 5:
                raise TypeError("The data callback must take five arguments. "
                                f"Given callback takes {num_params}.")
        self.__data_callback = data_callback

        # Check and set running conditions.
        self.__run_forever = run_forever
        if condition is not None:
            if not callable(condition):
                raise TypeError("The condition must be a callable.")
            num_params = len(inspect.signature(condition).parameters)
            if num_params != 5:
                raise TypeError("The condition must take five arguments. "
                                f"Given callable takes {num_params}.")
        self.__condition = condition

    def reset(self) -> None:
        """
        Reset the controller state.

        If the controller is running, pause execution,
        reset the controller, then resume execution.
        """
        if self.__running.is_set():
            # Only allow one thread to reset the controller at a time.
            with self.__atomic_update_lock:
                self.__stop()
                # Wait for the controller to stop before resetting.
                with self.__lock:
                    self.__system_controller.reset()
                self.__start()
        else:
            self.__system_controller.reset()

    def stop(self) -> None:
        """
        Stop the controller.

        Do nothing if the controller is not running.

        Raises
        ------
        `RuntimeError` - If the controller is running in a context manager.
        """
        if self.__in_context:
            raise RuntimeError("Cannot directly stop an AutoSystemController "
                               "while it is running in a context manager.")
        self.__stop()

    @contextmanager
    def context_run(
        self,
        tick_rate: int = 10,
        time_factor: float = 1.0,
        data_callback: DataCallback = None,
        reset: bool = True
    ) -> Iterator[None]:
        """
        Start the controller in a with-statement context.

        The controller is stopped automatically when the context is exited.

        For example:
        ```
        with controller.start(tick_rate, time_factor, reset):
            # Do stuff concurrently...
        ```
        Is equivalent to:
        ```
        try:
            controller.run_forever(tick_rate, time_factor)
            # Do stuff concurrently...
        finally:
            controller.stop(reset)
        ```

        Parameters
        ----------
        `tick_rate: int = 10` - The tick rate of the controller (in ticks per
        second). This is approximate, the actual tick rate may vary, the only
        guarantee is that the tick rate will not exceed the given value.

        `time_factor: float = 1.0` - The time factor to use when calculating
        time differences. The tick rate is multiplied by this value to get the
        tick rate relative to the time factor.

        `data_callback: (int, float, float, float, float) -> None = None` - A
        callable to callback control data to. The function parameters are:
        `(iteration, error, output, delta time, total time)`.

        `reset: bool = True` - Whether to also reset the controller state when
        the context is exited.

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

    def run_forever(
        self,
        tick_rate: int = 10,
        time_factor: float = 1.0,
        data_callback: DataCallback = None
    ) -> None:
        """
        Run the controller in a seperate thread until a stop call is made.

        This is a non-blocking call.

        Parameters
        ----------
        `tick_rate: int = 10` - The tick rate of the controller (in ticks per
        second). This is approximate, the actual tick rate may vary, the only
        guarantee is that the tick rate will not exceed the given value.

        `time_factor: float = 1.0` - The time factor to use when calculating
        time differences. The tick rate is multiplied by this value to get the
        tick rate relative to the time factor.

        `data_callback: (int, float, float, float, float) -> None = None` - A
        callable to callback control data to. The function parameters are:
        `(iteration, error, output, delta time, total time)`.

        Raises
        ------
        `RuntimeError` - If the controller is already running.
        """
        with self.__atomic_update_lock:
            self.__check_running_state()
            self.__set_parameters(
                tick_rate,
                time_factor,
                data_callback,
                True,
                None
            )
            self.__start()

    def run_for(
        self,
        max_ticks: Optional[int] = None,
        max_time: Optional[Real] = None,
        tick_rate: int = 10,
        time_factor: float = 1.0,
        data_callback: DataCallback = None
    ) -> None:
        """
        Run the controller for a given number of ticks or amount of time.

        The controller stops when either the tick or time limit is reached.

        Parameters
        ----------
        `max_ticks: int` - The maximum number of ticks to run for.

        `max_time: float` - The maximum amount of time in seconds to run for.

        `tick_rate: int = 10` - The tick rate of the controller (in ticks per
        second). This is approximate, the actual tick rate may vary, the only
        guarantee is that the tick rate will not exceed the given value.

        `time_factor: float = 1.0` - The time factor to use when calculating
        time differences. The tick rate is multiplied by this value to get the
        tick rate relative to the time factor.

        `data_callback: (int, float, float, float, float) -> None = None` - A
        callable to callback control data to. The function parameters are:
        `(iteration, error, output, delta time, total time)`.

        Raises
        ------
        `RuntimeError` - If the controller is already running.
        """
        with self.__atomic_update_lock:
            self.__check_running_state()
            run_forever: bool = False

            if max_ticks is not None and max_time is not None:
                def condition(i, e, o, dt, tt):
                    return i < max_ticks and tt < max_time
                run_forever = True
            elif max_ticks is not None:
                def condition(i, e, o, dt, tt):
                    return i < max_ticks
                run_forever = True
            elif max_time is not None:
                def condition(i, e, o, dt, tt):
                    return tt < max_time
                run_forever = True

            self.__set_parameters(
                tick_rate,
                time_factor,
                data_callback,
                run_forever,
                condition
            )
            self.__start()

    def run_while(
        self,
        condition: Condition,
        tick_rate: int = 10,
        time_factor: float = 1.0,
        data_callback: DataCallback = None
    ) -> None:
        """
        Run the controller while a condition is true.

        Parameters
        ----------
        `condition: (int, float, float, float) -> bool` - The
        condition to test. The function signature is
        `(iterations, error, output, delta time, total time) -> bool`,
        the controller will stop when the condition returns `False`.

        `tick_rate: int = 10` - The tick rate of the controller (in ticks per
        second). This is approximate, the actual tick rate may vary, the only
        guarantee is that the tick rate will not exceed the given value.

        `time_factor: float = 1.0` - The time factor to use when calculating
        time differences. The tick rate is multiplied by this value to get the
        tick rate relative to the time factor.

        `data_callback: (int, float, float, float, float) -> None = None` - A
        callable to callback control data to. The function parameters are:
        `(iteration, error, output, delta time, total time)`.

        Raises
        ------
        `RuntimeError` - If the controller is already running.
        """
        with self.__atomic_update_lock:
            self.__check_running_state()
            self.__set_parameters(
                tick_rate,
                time_factor,
                data_callback,
                False,
                condition
            )
            self.__start()
