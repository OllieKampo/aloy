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

__all__ = ("InvertedPendulumSystem",)

from dataclasses import dataclass
import math
from control.controllers import ControlledSystem

@dataclass
class InvertedPendulumSystem(ControlledSystem):
    """A non-linear inverted pendulum system."""
    
    ## Physical properties of the system.
    pendulum_length: float = 1.0
    gravity: float = 9.80665
    steady_state_error: float = 1.0
    
    ## Current state of the system.
    force: float = 0.0
    pendulum_angle: float = math.pi / 4.0
    pendulum_angular_velocity: float = 0.0
    
    def reset(self) -> None:
        """Reset the system to its initial state."""
        self.force = 0.0
        self.pendulum_angle = math.pi / 4.0
        self.pendulum_angular_velocity = 0.0
    
    def update_system(self, delta_time: float) -> None:
        """Update the system state."""
        cart_acc = (self.force - self.steady_state_error)
        pend_ang_acc = (((-cart_acc * math.cos(self.pendulum_angle))
                         + (self.gravity * math.sin(self.pendulum_angle)))
                        / self.pendulum_length)
        
        v0 = self.pendulum_angular_velocity
        self.pendulum_angular_velocity += pend_ang_acc * delta_time
        self.pendulum_angle += ((v0 + self.pendulum_angular_velocity) / 2.0) * delta_time
        
        max_angle = (math.pi * (5.0 / 12.0))
        if self.pendulum_angle > max_angle:
            self.pendulum_angle = max_angle
            self.pendulum_angular_velocity = 0.0
        elif self.pendulum_angle < -max_angle:
            self.pendulum_angle = -max_angle
            self.pendulum_angular_velocity = 0.0
    
    def error_variables(self) -> dict[str, str]:
        return {"pendulum_angle": "force"}
    
    def get_control_input(self, var_name: str | None = None) -> float:
        return self.pendulum_angle
    
    def get_setpoint(self, var_name: str | None = None) -> float:
        return 0.0
    
    def set_control_output(self, output: float, delta_time: float, var_name: str | None = None) -> None:
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
        cart_acc = (self.force - self.steady_state_error) / self.cart_mass
        pend_ang_acc = (((-cart_acc * math.cos(self.pendulum_angle))
                         + (self.gravity * math.sin(self.pendulum_angle)))
                        / self.pendulum_length)
        
        v0 = self.pendulum_angular_velocity
        self.pendulum_angular_velocity += pend_ang_acc * delta_time
        self.pendulum_angle += ((v0 + self.pendulum_angular_velocity) / 2.0) * delta_time
        
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
    
    def error_variables(self) -> dict[str, str]:
        return {"pendulum_angle": "force", "cart_position": "force"}
    
    def control_variables(self) -> dict[str, str]:
        return {"force": "force"}

    def get_control_input(self, var_name: str) -> float:
        if var_name == "pendulum_angle":
            return self.pendulum_angle
        elif var_name == "cart_position":
            return self.cart_position
        else:
            raise ValueError(f"Invalid variable name: {var_name}")
    
    def get_setpoint(self, var_name: str) -> float:
        return 0.0

    def set_control_output(self, output: float, delta_time: float, var_name: str) -> None:
        self.force = output
        self.update_system(delta_time)
