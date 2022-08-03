###########################################################################
###########################################################################
## A general implementation of a particle swarm optimisation algorithm.  ##
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

"""General implementation of a particle swarm optimisation algorithm."""

import dataclasses
from fractions import Fraction
import math
from numbers import Real
from typing import Callable, Literal, Optional, Sequence
from matplotlib import pyplot as plt

import numpy as np
import numpy.typing as npt

from datahandling.makedata import DataHolder

@dataclasses.dataclass(frozen=True)
class Dimension:
    """
    A dimension of the search space.
    
    Fields
    ------
    `lower_bound: float` - The lower bound of the domain of the dimension.
    
    `upper_bound: float` - The upper bound of the domain of the dimension.
    
    `max_velocity: float` - The maximum velocity of a particle moving along the dimension.
    """
    
    lower_bound: Real
    upper_bound: Real
    max_velocity: Real

class OptimizationSolution:
    __slots__ = ()
    
    

@dataclasses.dataclass(frozen=True)
class ParticleSwarmSolution(OptimizationSolution):
    """
    Represents the solution of a particle swarm optimisation algorithm.
    
    Elements
    --------
    `best_position: npt.NDArray[np.float64]` - The best position of the swarm.
    
    `best_fitness: np.float64` - The best fitness of the swarm.
    """
    
    ## Particles
    best_position: npt.NDArray[np.float64]
    best_fitness: np.float64
    positions: npt.NDArray[np.float64]
    fitnesses: npt.NDArray[np.float64]
    
    ## Loop variables
    total_iterations: int
    stagnated_at_iteration: int
    
    ## Stop conditions
    iterations_limit_reached: bool = False
    stagnation_limit_reached: bool = False
    fitness_limit_reached: bool = False
    
    def __str__(self) -> str:
        """Return a decscription of the best solution obtained and the stop condition that was reached."""
        best_parameters = ", ".join(f"{param:.4f}" for param in self.best_position)
        stop_condition: str = ""
        if self.iterations_limit_reached:
            stop_condition = "iterations limit reached"
        if self.stagnation_limit_reached:
            stop_condition = "stagnation limit reached"
            stop_condition += f" (stagnated at iteration {self.stagnated_at_iteration})"
        if self.fitness_limit_reached:
            stop_condition = "fitness limit reached"
        return f"PSO Solution :: best parameters = [{best_parameters}], best fitness = {self.best_fitness}, " \
               f"total iterations = {self.total_iterations}, stop condition = {stop_condition}"

class ParticleSwarmSystem:
    """Particle swarm optimisation algorithm."""
    
    __slots__ = ("__dimensions",
                 "__fitness_function",
                 "__maximise")
    
    def __init__(self,
                 dimensions: Sequence[Dimension],
                 fitness_function: Callable[[npt.NDArray[np.float64]], np.float64],
                 maximise: bool = True
                 ) -> None:
        """
        Create a particle swarm optimisation system from a list of dimensions and an objective function.
        
        A dimension must be added for each variable of the objective function (i.e. the search space).
        """
        self.__dimensions: list[Dimension] = list(dimensions)
        self.__fitness_function: Callable[[npt.NDArray[np.float64]], np.float64] = fitness_function
        self.__maximise: bool = maximise
    
    @staticmethod
    def calculate_decay(iteration: int,
                        iterations_limit: int,
                        decay_start: int,
                        decay_end: int,
                        initial_value: Fraction,
                        decay_constant: Fraction,
                        decay_type: Literal["lin", "pol", "exp"] = "lin"
                        ) -> Fraction:
        """Calculate the decay of the particle inertia or personal coefficient."""
        ## Large decay constant should always result in slower decay.
        ## Decayer should check if the decay constant is correct for the given type.
        
        decay_limit: int = min(iterations_limit, decay_end) - 1
        decay_iterations: int = min(decay_limit - decay_start, max(0, iteration - decay_start))
        
        if decay_type == "lin":
            return max(Fraction(0.0), initial_value - initial_value * (decay_constant * decay_iterations))
        
        elif decay_type == "pol":
            return initial_value * (decay_constant ** decay_iterations)
        
        elif decay_type == "exp":
            return initial_value * math.exp(-((1.0 - decay_constant) * decay_iterations))
        
        elif decay_type == "exp-auto":
            if decay_iterations < decay_constant * iterations_limit:
                return initial_value * (1.0 - (math.log(decay_iterations) / math.log(decay_constant * iterations_limit)))
            return Fraction(0.0)
        
        elif decay_type == "sin":
            if decay_iterations < decay_constant * iterations_limit:
                return initial_value * ((math.cos(math.pi * (decay_iterations / (decay_constant * iterations_limit))) / 2.0) + 0.5)
            return Fraction(0.0)
    
    def run(self,
            total_particles: int,
            
            ## Stop criteria
            iterations_limit: Optional[int] = 1000,
            stagnation_limit: Optional[int | float | Fraction] = 0.1,
            fitness_limit: Optional[Real] = None,
            
            ## Particle inertia
            inertia: Fraction = Fraction(1.0),
            inertia_decay: Optional[Fraction] = None,
            inertia_decay_type: Literal["lin", "pol", "exp"] = "lin",
            
            ## Particle direction of movement coefficients
            personal_coefficient: Fraction = Fraction(1.0),
            coefficient_decay: Optional[Fraction] = None,
            coefficient_decay_type: Optional[Literal["lin", "pol", "exp"]] = "lin"
            
            ) -> Optional[ParticleSwarmSolution]:
        """
        Run the particle swarm optimisation algorithm.
        
        `personal_coefficient: Fraction` - 
        
        `local_coefficient: Fraction` - 
        
        `global_coefficient: Fraction` - The global (or social) coefficient used in particle velocity updates.
        A high global coefficient will promote an exploitative search towards the global best position.
        If not given or None, the global coefficient is 1.0 minus the personal coefficient.
        
        `coefficient_decay: {Fraction | None}` - The decay rate of the local coefficient.
        The respective gain rate of the global coefficient is 1.0 minus the local coefficient.
        A high decay constant will cause rapid decay of the local coefficient and gain in the global coefficient.
        Conversely, a low decay constant will cause slow decay of the local coefficient and gain in the global coefficient.
        If not given or None, then the decay rate is the ratio of the personal coefficient to the iterations limit.
        
        `coefficient_decay_type: {Literal["lin", "pol", "exp", "hyp"] | None}` - The decay rate type of the coefficient decay.
        Linear decay redcues the coefficient linearly with iterations, i.e. reduces by a constant value on each iteration.
        Exponential decay reduces the coefficient exponentially with iterations, i.e. reduces by a multiple on each iteration.
        Exponential decay causes a larger changes in the coefficients during early iterations of search than linear decay.
        If not given or None, then coefficient decay is disabled.
        """
        if total_particles < 1:
            raise ValueError(f"Total particles must be at least 1. Got; {total_particles}")
        
        if all(map(lambda x: x is None, [iterations_limit, stagnation_limit, fitness_limit])):
            raise ValueError("At least one of the stop conditions must be given and not None.")
        
        if isinstance(stagnation_limit, (float, Fraction)):
            stagnation_limit = int(stagnation_limit * iterations_limit)
        
        initial_inertia = Fraction(inertia)
        inertia = initial_inertia
        initial_personal_coefficient = Fraction(personal_coefficient)
        personal_coefficient = initial_personal_coefficient
        global_coefficient = Fraction(1.0) - personal_coefficient
        
        if inertia_decay is None:
            inertia_decay = Fraction(inertia / iterations_limit)
        elif not (Fraction(0.0) <= inertia_decay <= Fraction(1.0)):
            raise ValueError(f"Inertia decay constant be between 0.0 and 1.0. Got; {inertia_decay}")
        if coefficient_decay is None:
            coefficient_decay = Fraction(personal_coefficient / iterations_limit)
        elif not (Fraction(0.0) <= coefficient_decay <= Fraction(1.0)):
            raise ValueError(f"Coefficient decay constant be between 0.0 and 1.0. Got; {coefficient_decay}")
        
        ## Random number generator
        rng = np.random.default_rng()
        
        ## Array of dimensions: (lower_bound, upper_bound, max_velocity)
        dimensions = np.array([dataclasses.astuple(dimension) for dimension in self.__dimensions])
        
        ## Create positions and velocities for each particle:
        ##   - Arrays have row for each particle and column for each dimension.
        position_vectors = rng.uniform(dimensions[:,0], dimensions[:,1], (total_particles, dimensions.shape[0]))
        velocity_vectors = rng.uniform(-dimensions[:,2], dimensions[:,2], (total_particles, dimensions.shape[0]))
        
        ## Evaluate fitness of each particle
        particles_fitness = np.apply_along_axis(self.__fitness_function, 1, position_vectors)
        
        ## Personal best fitness and position for each particle (initialise to current fitness)
        best_particles_fitness = particles_fitness.copy()
        best_position_vectors = position_vectors.copy()
        
        ## Global best fitness and position over all particles
        if self.__maximise:
            global_best_index = np.argmax(particles_fitness)
        else: global_best_index = np.argmin(particles_fitness)
        global_best_fitness = particles_fitness[global_best_index]
        global_best_position = position_vectors[global_best_index]
        
        ## Loop variables
        iterations: int = 0
        stagnated_iterations: int = 0
        stagnated_at: int = 0
        
        ## Stop conditions
        iteration_limit_reached: bool = False
        stagnation_limit_reached: bool = False
        fitness_limit_reached: bool = False
        
        ## Data structure for storing history of best fitness and position.
        dataholder = DataHolder(["iteration",
                                 "global_best_fitness", "global_best_position",
                                 "personal_coefficient", "global_coefficient"],
                                converters={"personal_coefficient" : float,
                                            "global_coefficient" : float})
        
        ## Iterate until some stop condition is reached.
        while not ((iterations_limit is not None
                    and (iteration_limit_reached := iterations >= iterations_limit))
                   or (stagnation_limit is not None
                       and (stagnation_limit_reached := stagnated_iterations >= stagnation_limit))
                   or (fitness_limit is not None
                       and (fitness_limit_reached := global_best_fitness >= fitness_limit))):
            
            if iterations != 0:
                if inertia_decay_type is not None:
                    inertia = self.calculate_decay(iterations,
                                                   iterations_limit,
                                                   initial_inertia,
                                                   inertia_decay,
                                                   inertia_decay_type)
                
                if coefficient_decay_type is not None:
                    personal_coefficient = self.calculate_decay(iterations,
                                                                iterations_limit,
                                                                initial_personal_coefficient,
                                                                coefficient_decay,
                                                                coefficient_decay_type)
                    global_coefficient = Fraction(1.0) - personal_coefficient
            
            ## For each particle, generate two random numbers for each dimension;
            ##      - one for the personal best, and one for the global best.
            update_vectors = rng.random((total_particles, 2, dimensions.shape[0]))
            
            ## Velocity is updated based on inertia and random contribution of displacement from;
            ##      - personal best position,
            ##      - local "friend" group position,
            ##      - global best position.
            personal_displacements = best_position_vectors - position_vectors
            global_displacements = global_best_position - position_vectors
            
            velocity_vectors = (inertia * velocity_vectors
                                + (personal_coefficient * update_vectors[:,0] * personal_displacements)
                                + (global_coefficient * update_vectors[:,1] * global_displacements))
            velocity_vectors = np.maximum(np.minimum(velocity_vectors, dimensions[:, 2]), -dimensions[:, 2])
            
            ## Position is updated based on previous position and velocity:
            ##    - take the maximum of the lower bound of each dimension and the minimum of upper bound of each dimension and the new position plus velocity.
            position_vectors = np.maximum(np.minimum(position_vectors + velocity_vectors, dimensions[:, 1]), dimensions[:, 0])
            
            ## Evaluate fitness of each particle and update personal best.
            particles_fitness = np.apply_along_axis(self.__fitness_function, 1, position_vectors)
            
            if self.__maximise:
                particles_better_than_best = particles_fitness >= best_particles_fitness
            else: particles_better_than_best = particles_fitness < best_particles_fitness
            best_particles_fitness[particles_better_than_best] = particles_fitness[particles_better_than_best]
            best_position_vectors[particles_better_than_best] = position_vectors[particles_better_than_best]
            
            ## Global best fitness and position over all particles.
            if self.__maximise:
                current_best_index = np.argmax(particles_fitness)
            else: current_best_index = np.argmin(particles_fitness)
            current_best_fitness = particles_fitness[current_best_index]
            if ((self.__maximise and current_best_fitness > global_best_fitness)
                or (not self.__maximise and current_best_fitness < global_best_fitness)):
                global_best_fitness = current_best_fitness
                global_best_position = position_vectors[current_best_index]
                stagnated_iterations = 0
                stagnated_at = iterations
            else: stagnated_iterations += 1
            iterations += 1
            
            dataholder.add_row([iterations,
                                global_best_fitness, global_best_position,
                                personal_coefficient, global_coefficient])
        
        return (ParticleSwarmSolution(global_best_position, global_best_fitness,
                                      position_vectors, particles_fitness,
                                      iterations, stagnated_at,
                                      iteration_limit_reached, stagnation_limit_reached, fitness_limit_reached),
                dataholder.to_dataframe())
