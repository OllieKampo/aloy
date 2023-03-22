###########################################################################
###########################################################################
## Module defining various benchmark control systems.                    ##
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

"""Module defining various benchmark control systems."""

__all__ = (
    "DynamicSystem",
    "InvertedPendulumSystem",
    "CartAndPendulumSystem",
    "MassSpringDamperSystem",
    "BoilerSystem"
)

from abc import abstractmethod, abstractstaticmethod
from dataclasses import dataclass
import math
from typing_extensions import override

import numpy as np
from control.controllers import ControlledSystem


# @dataclass
# class Historian:
#     """Stores the state of a system over time."""

#     def __init__(self, system: "DynamicSystem") -> None:
#         """Initialize the historian."""
#         self.system = system
#         self.state_variables = system.state_variables()
#         self.time = 0.0
#         self.history = {var: [system.get_state()[i]]
#                         for i, var in enumerate(self.state_variables)}

#     def update(self, delta_time: float) -> None:
#         """Update the historian."""
#         self.time.append(self.time[-1] + delta_time)
#         state = self.system.get_state()
#         for i, var in enumerate(self.state_variables):
#             self.history[var].append(state[i])


class DynamicSystem:
    """Base class for dynamic systems."""

    @abstractstaticmethod
    def random_start(*args, **kwargs) -> "DynamicSystem":
        """Return a random initial state."""
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Reset the system to its initial state."""
        raise NotImplementedError

    @abstractmethod
    def update_system(self, delta_time: float) -> None:
        """Update the system state."""
        raise NotImplementedError


# class SimulatableDynamicSystem(DynamicSystem):
#     """Base class for dynamic systems that can be simulated."""

#     @abstractmethod
#     def state_variables(self) -> tuple[str]:
#         """Return the names of the state variables."""
#         raise NotImplementedError

#     @abstractmethod
#     def get_state(self) -> tuple[float]:
#         """Return the state of the system."""
#         raise NotImplementedError

#     @abstractmethod
#     def simulate(self, time: float, steps: int, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
#         """Simulate the system for a given amount of time."""
#         raise NotImplementedError


@dataclass
class InvertedPendulumSystem(DynamicSystem, ControlledSystem):
    """A non-linear inverted pendulum system."""

    ## Physical properties of the system.
    pendulum_length: float = 1.0
    gravity: float = 9.80665
    steady_state_error: float = -1.0

    ## Initial state of the system.
    initial_pendulum_angle: float = math.pi / 4.0
    initial_pendulum_angular_velocity: float = 0.0

    ## Current state of the system.
    force: float = 0.0
    pendulum_angle: float = math.pi / 4.0
    pendulum_angular_velocity: float = 0.0

    def reset(self) -> None:
        """Reset the system to its initial state."""
        self.force = 0.0
        self.pendulum_angle = self.initial_pendulum_angle
        self.pendulum_angular_velocity = self.initial_pendulum_angular_velocity

    def update_system(self, delta_time: float) -> None:
        """Update the system state."""
        cart_acc = (self.force + self.steady_state_error)
        pend_ang_acc = (((-cart_acc * math.cos(self.pendulum_angle))
                         + (self.gravity * math.sin(self.pendulum_angle)))
                        / self.pendulum_length)

        v0 = self.pendulum_angular_velocity
        self.pendulum_angular_velocity += pend_ang_acc * delta_time
        self.pendulum_angle += \
            ((v0 + self.pendulum_angular_velocity) / 2.0) * delta_time

        max_angle = (math.pi * (5.0 / 12.0))
        if self.pendulum_angle > max_angle:
            self.pendulum_angle = max_angle
            self.pendulum_angular_velocity = 0.0
        elif self.pendulum_angle < -max_angle:
            self.pendulum_angle = -max_angle
            self.pendulum_angular_velocity = 0.0

    @override
    @property
    def input_variables(self) -> tuple[str] | None:
        return ("pendulum_angle",)

    @override
    @property
    def output_variables(self) -> tuple[str] | None:
        return ("force",)

    @override
    def get_control_input(self, var_name: str | None = None) -> float:
        return self.pendulum_angle

    @override
    def get_setpoint(self, var_name: str | None = None) -> float:
        return 0.0

    @override
    def set_control_output(
        self,
        output: float,
        delta_time: float,
        var_name: str | None = None
    ) -> None:
        self.force = output
        self.update_system(delta_time)


@dataclass
class CartAndPendulumSystem(InvertedPendulumSystem):
    """A non-linear cart and pendulum system."""

    ## Physical properties of the system.
    cart_mass: float = 1.0

    ## Current state of the system.
    cart_position: float = 0.0
    cart_velocity: float = 0.0

    def reset(self) -> None:
        """Reset the system to its initial state."""
        self.force = 0.0
        self.cart_position = 0.0
        self.cart_velocity = 0.0
        self.pendulum_angle = math.pi / 4.0
        self.pendulum_angular_velocity = 0.0

    def update_system(self, delta_time: float) -> None:
        """Update the system state."""
        cart_acc = (self.force + self.steady_state_error) / self.cart_mass
        pend_ang_acc = (((-cart_acc * math.cos(self.pendulum_angle))
                         + (self.gravity * math.sin(self.pendulum_angle)))
                        / self.pendulum_length)

        v0 = self.pendulum_angular_velocity
        self.pendulum_angular_velocity += pend_ang_acc * delta_time
        self.pendulum_angle += \
            ((v0 + self.pendulum_angular_velocity) / 2.0) * delta_time

        max_angle = (math.pi * (5.0 / 12.0))
        if self.pendulum_angle > max_angle:
            self.pendulum_angle = max_angle
            self.pendulum_angular_velocity = 0.0
        elif self.pendulum_angle < -max_angle:
            self.pendulum_angle = -max_angle
            self.pendulum_angular_velocity = 0.0

        v0 = self.cart_velocity
        self.cart_velocity += cart_acc * delta_time
        self.cart_position += ((v0 + self.cart_velocity) / 2.0) * delta_time

    @override
    @property
    def input_variables(self) -> tuple[str] | None:
        return super().input_variables + ("cart_position",)

    @override
    def get_control_input(self, var_name: str) -> float:
        if var_name == "pendulum_angle":
            return self.pendulum_angle
        elif var_name == "cart_position":
            return self.cart_position
        else:
            raise ValueError(f"Invalid variable name: {var_name}")

    @override
    def set_control_output(
        self,
        output: float,
        delta_time: float,
        var_name: str | None = None
    ) -> None:
        self.force = output
        self.update_system(delta_time)


@dataclass
class MassSpringDamperSystem(DynamicSystem, ControlledSystem):
    """A non-linear mass-spring-damper system."""

    ## Physical properties of the system.
    mass: float = 2.5
    spring_constant: float = 1.0
    damping_constant: float = 1.0
    steady_state_error: float = 0.0

    ## Initial state of the system.
    initial_force: float = 0.0
    initial_position: float = 0.0
    initial_velocity: float = 0.0

    ## Current state of the system.
    force: float = 0.0
    position: float = 0.0
    velocity: float = 0.0
    acceleration: float = 0.0

    def __post_init__(self) -> None:
        self.reset()

    @staticmethod
    def random_start(
        mass: float = 2.5,
        spring_constant: float = 1.0,
        damping_constant: float = 1.0,
        steady_state_error: float = 0.0,
        force_range: tuple[float, float] = (-1.0, 1.0),
        position_range: tuple[float, float] = (-5.0, 5.0),
        velocity_range: tuple[float, float] = (-2.5, 2.5),
        seed: int | None = None
    ) -> "MassSpringDamperSystem":
        """Create a mass-spring-damper system with random initial state."""
        rng = np.random.default_rng(seed)
        return MassSpringDamperSystem(
            mass=mass,
            spring_constant=spring_constant,
            damping_constant=damping_constant,
            steady_state_error=steady_state_error,
            initial_force=rng.uniform(*force_range),
            initial_position=rng.uniform(*position_range),
            initial_velocity=rng.uniform(*velocity_range)
        )

    def reset(self) -> None:
        """Reset the system to its initial state."""
        self.force = self.initial_force
        self.position = self.initial_position
        self.velocity = self.initial_velocity

    def update_system(self, delta_time: float) -> None:
        """Update the system state."""
        acc = ((self.force + self.steady_state_error)
               - (self.spring_constant * self.position)
               - (self.damping_constant * self.velocity)) / self.mass
        self.acceleration = acc
        vel = self.velocity
        self.velocity += acc * delta_time
        self.position += ((vel + self.velocity) / 2.0) * delta_time

    @property
    def input_variables(self) -> tuple[str] | None:
        return ("position",)

    @property
    def output_variables(self) -> tuple[str] | None:
        return ("velocity",)

    def get_control_input(self, var_name: str | None = None) -> float:
        return self.position

    def get_setpoint(self, var_name: str | None = None) -> float:
        return 0.0

    def set_control_output(
        self,
        output: float,
        delta_time: float,
        var_name: str | None = None
    ) -> None:
        self.force = output
        self.update_system(delta_time)


@dataclass
class BoilerSystem(DynamicSystem, ControlledSystem):
    """A non-linear boiler system."""

    ## Physical properties of the system.
    fluid_capacity: float = 0.5  # m^3
    fluid_density: float = 1000.0  # kg/m^3
    fluid_specific_heat_capacity: float = 4184.0  # J/(kg*K)

    heater_efficiency: float = 0.65  # W/W
    heater_heat_capacity: float = 2500.0  # J/K
    heater_heat_transfer_coefficient: float = 0.85  # W/(m^2*K)

    boiler_heat_transfer_coefficient: float = 0.15  # W/(m^2*K)
    ambient_temperature: float = 21.3  # K

    ## Initial state of the system.
    initial_heater_power: float = 0.0  # W
    initial_heater_temperature: float = 21.3  # K
    initial_fluid_temperature: float = 21.3  # K

    ## Setpoints for the system.
    temperature_setpoint: float = 100.0  # K

    ## Current state of the system.
    heater_power: float = 0.0  # W
    heater_temperature: float = 0.0  # K
    fluid_temperature: float = 0.0  # K

    def __post_init__(self) -> None:
        self.reset()

    @staticmethod
    def random_start(
        fluid_capacity: float = 10.0,
        fluid_density: float = 1.0,
        fluid_specific_heat_capacity: float = 1.0,
        heater_efficiency: float = 1.0,
        heater_heat_capacity: float = 1.0,
        heater_heat_transfer_coefficient: float = 0.85,
        boiler_heat_transfer_coefficient: float = 0.15,
        ambient_temperature: float = 1.0,
        temperature_setpoint: float = 1.0,
        heater_power_range: tuple[float, float] = (0.0, 1.0),
        heater_temperature_range: tuple[float, float] = (0.0, 1.0),
        fluid_temperature_range: tuple[float, float] = (0.0, 1.0),
        seed: int | None = None
    ) -> "BoilerSystem":
        """Create a boiler system with random initial state."""
        rng = np.random.default_rng(seed)
        return BoilerSystem(
            fluid_capacity=fluid_capacity,
            fluid_density=fluid_density,
            fluid_specific_heat_capacity=fluid_specific_heat_capacity,
            heater_efficiency=heater_efficiency,
            heater_heat_capacity=heater_heat_capacity,
            heater_heat_transfer_coefficient=heater_heat_transfer_coefficient,
            boiler_heat_transfer_coefficient=boiler_heat_transfer_coefficient,
            ambient_temperature=ambient_temperature,
            temperature_setpoint=temperature_setpoint,
            initial_heater_power=rng.uniform(*heater_power_range),
            initial_heater_temperature=rng.uniform(*heater_temperature_range),
            initial_fluid_temperature=rng.uniform(*fluid_temperature_range)
        )

    def reset(self) -> None:
        """Reset the system to its initial state."""
        self.heater_power = self.initial_heater_power
        self.heater_temperature = self.initial_heater_temperature
        self.fluid_temperature = self.initial_fluid_temperature

    def update_system(self, delta_time: float) -> None:
        """Update the system state."""
        heater_temperature = self.heater_temperature
        fluid_temperature = self.fluid_temperature

        heater_temperature += ((self.heater_power * self.heater_efficiency)
                               * delta_time) / self.heater_heat_capacity
        fluid_temperature -= ((fluid_temperature - self.ambient_temperature)
                              * self.boiler_heat_transfer_coefficient
                              * delta_time)
        energy_transfer = ((heater_temperature - fluid_temperature)
                           * self.heater_heat_transfer_coefficient
                           * delta_time)
        fluid_temperature += (energy_transfer
                              / (self.fluid_specific_heat_capacity
                                 * self.fluid_density
                                 * self.fluid_capacity))
        heater_temperature -= energy_transfer / self.heater_heat_capacity

        self.heater_temperature = heater_temperature
        self.fluid_temperature = fluid_temperature

    @property
    def input_variables(self) -> tuple[str] | None:
        return ("fluid_temperature",)

    @property
    def output_variables(self) -> tuple[str] | None:
        return ("temperature_setpoint",)

    def get_control_input(self, var_name: str | None = None) -> float:
        return self.fluid_temperature

    def get_setpoint(self, var_name: str | None = None) -> float:
        return self.temperature_setpoint

    def set_control_output(
        self,
        output: float,
        delta_time: float,
        var_name: str | None = None
    ) -> None:
        self.update_system(delta_time)
        self.heater_power = output
