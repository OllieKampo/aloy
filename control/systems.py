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
    cart_mass: float = 1.0
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
    
    def get_error(self, var_name: str | None = None) -> float:
        """Get the current error from the setpoint(s) for the control system."""
        return self.pendulum_angle
    
    def set_output(self, output: float, delta_time: float, var_name: str | None = None) -> None:
        """Set the control output(s) to the system."""
        self.force = output
        self.update_system(delta_time)
